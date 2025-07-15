import os
import yaml
import torch
from datasets import load_dataset, IterableDataset, Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, get_scheduler
from accelerate import Accelerator
from huggingface_hub import HfFolder, create_repo, upload_folder
import wandb
import time
import torch.nn.functional as F
from galore_torch import GaLoreAdamW8bit
from transformers import TrainerCallback
from itertools import islice
from huggingface_hub import login
import gc

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_environment(config):
    # os.environ["WANDB_PROJECT"] = config.get("wandb_project", "default_project")
    # os.environ["WANDB_ENTITY"] = config.get("wandb_entity", "default_entity")
    # wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"])
    os.environ["WANDB_DISABLED"] = 'true'
    return Accelerator()

def load_and_preprocess_dataset(config, student_tokenizer):
    def tokenize_function(examples):
        return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

    datasets = []
    for subset in config["dataset"]["subsets"]:
        # Load the dataset as an IterableDataset
        dataset = load_dataset(
            config["dataset"]["name"],
            subset=subset,
            split='split',
            streaming=True
            # Downloading then mapping the dataset takes lot of time, so we use streaming
        )

        # Keep only the 'text' column for all subsets
        if 'text' in dataset.column_names:
            dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
        else:
            raise ValueError(f"The 'text' column is missing in the {subset} subset.")
        datasets.append(dataset)

    # Concatenate all datasets
    full_dataset = concatenate_datasets(datasets)

    # Create evaluation dataset (first N examples)
    eval_dataset = Dataset.from_list(list(islice(full_dataset, config["dataset"]["eval_samples"])))
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Create training dataset (skip first N examples)
    def generate_train_examples():
        for i, example in enumerate(full_dataset):
            if i >= config["dataset"]["eval_samples"]:
                yield example

    train_dataset = IterableDataset.from_generator(generate_train_examples)
    train_dataset = train_dataset.map(
        tokenize_function,
        remove_columns=train_dataset.column_names
    )

    return train_dataset, eval_dataset

def load_models_and_tokenizers(config):
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"model_kwargs: {model_kwargs}")

    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"], add_eos_token=True)
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"], add_eos_token=True)

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
        print(f"Set pad_token to eos_token: {student_tokenizer.pad_token}")

    teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

    teacher_model.eval() # set teacher model to evaluation mode

    return teacher_model, student_model, teacher_tokenizer, student_tokenizer

def pad_logits(student_logits, teacher_logits):
    student_size = student_logits.size(-1)
    teacher_size = teacher_logits.size(-1)

    if student_size == teacher_size:
        return student_logits, teacher_logits
    elif student_size < teacher_size:
        pad_tensor = torch.zeros((teacher_logits.shape[0], teacher_logits.shape[1], teacher_size - student_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        student_logits = torch.cat((student_logits, pad_tensor), dim=-1)
        return student_logits, teacher_logits
    else: # student_size > teacher_size
        pad_tensor = torch.zeros((student_logits.shape[0], student_logits.shape[1], student_size - teacher_size), dtype=student_logits.dtype, device=student_logits.device)
        teacher_logits = torch.cat((teacher_logits, pad_tensor), dim=-1)
        return student_logits, teacher_logits
    

class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config', None)
        self.teacher_model = kwargs.pop('teacher_model', None)
        super().__init__(*args, **kwargs)

        # Ensure teacher model is on the same device as the student model
        if self.teacher_model.device != self.model.device:
            self.teacher_model = self.teacher_model.to(self.model.device)

        # Ensure teacher model is in eval mode
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        if hasattr(model, 'module'):
            device = model.module.device
        else:
            device = next(model.parameters()).device

        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        # Check if 'labels' are in the inputs, if not, use 'input_ids' as labels
        labels = inputs.get('labels', inputs.get('input_ids'))

        if labels is None:
            raise ValueError("Neither 'labels' nor 'input_ids' found in inputs. Cannot compute loss.")

        custom_loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, labels)
        if return_outputs:
            return custom_loss, student_outputs
        return custom_loss
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        student_logits, teacher_logits = pad_logits(student_logits, teacher_logits)

        KL_loss = self.forward_kl_divergence(student_logits, teacher_logits)

        if self.config["distillation"]["alpha"] == 1:
            # Calculate the original loss (cross-entropy loss)
            original_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)
        else:
            original_loss = 0

        combined_loss = self.config["distillation"]["alpha"] * KL_loss + (1 - self.config["distillation"]["alpha"]) * original_loss
        return combined_loss


    def forward_kl_divergence(self, student_logits, teacher_logits):
        temperature = self.config["distillation"]["temperature"]
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        kl_div = F.kl_div(
            student_log_probs,
            teacher_log_probs.exp(),
            reduction='batchmean',
            log_target=False
        )

        return kl_div * (temperature ** 2) / self.config["tokenizer"]["max_length"]
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        
        output = super().evaluation_loop( dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        eval_loss = 0.0
        num_examples = 0
        chunk_size = 4 # Adjust this value based on your GPU memory

        for step, inputs in enumerate(dataloader):
            for i in range(0, inputs["input_ids"].size(0), chunk_size):
                chunk_inputs = {k: v[i:i+chunk_size] for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                loss = self.compute_loss(self.model, chunk_inputs)
                eval_loss += loss.detach().float() * len(chunk_inputs["input_ids"])
                num_examples += len(chunk_inputs["input_ids"])

        eval_loss /= num_examples
        output.metrics[f"{metric_key_prefix}_loss"] = eval_loss.item()
        return output
    

def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Max Allocated: {torch.cuda.max_cuda_memory_allocated() / 1e9:.2f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    print(f"Max Reserved: {torch.cuda.max_cuda_memory_reserved() / 1e9:.2f}GB")

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

class MemoryTracker(TrainerCallback):
    def __init__(self, print_every=100):
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0:
            print(f"step {state.global_step}:")
            print_memory_stats()
            clear_memory()

# def get_custom_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
#     return get_scheduler(
#         "constant",
#         optimizer=optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_training_steps,
#     )


def get_custom_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, initial_phase_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps # Linear warmup
        elif current_step < initial_phase_steps:
            return 1.0 # Constant learning rate for initial phase
        else:
            # Linear annealing for the remaining steps
            return (1.0 - (current_step - initial_phase_steps) / (num_training_steps - initial_phase_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)




def main(config_path):
    config = load_config(config_path)
    accelerator = setup_environment(config)

    teacher_model, student_model, teacher_tokenizer, student_tokenizer = load_models_and_tokenizers(config)

    print(f"Student model: {student_model}")
    print("Memory after loading models:")
    print_memory_stats()
    clear_memory()

    train_dataset, eval_dataset = load_and_preprocess_dataset(config, student_tokenizer)

    # Ensure train_dataset is iterable and eval_dataset is a regular dataset
    assert isinstance(train_dataset, IterableDataset)
    assert isinstance(eval_dataset, Dataset)

    # Calculate max_steps
    total_samples = config["dataset"]["total_train_samples"] - config["dataset"]["eval_samples"]
    batch_size = config["training"]["per_device_train_batch_size"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    num_gpus = torch.cuda.device_count()
    num_epochs = config["training_aux"]["num_train_epochs"]

    max_steps = int(total_samples / (batch_size * grad_accum_steps * num_gpus) * num_epochs)

    # Ensure max_steps is a positive integer
    max_steps = max(1, max_steps)

    initial_phase_steps = int(max_steps * (1 - config["training_aux"]["annealing_phase_fraction"]))

    # Calculate save_steps, logging_steps, and eval_steps
    save_steps = max(1, int(max_steps * config["training_aux"]["save_steps_fraction"]))
    logging_steps = max(1, int(max_steps * config["training_aux"]["logging_steps_fraction"]))
    eval_steps = max(1, int(max_steps * config["training_aux"]["eval_steps_fraction"]))

    # Calculate warmup steps if using warmup
    warmup_steps = int(max_steps * config["training"]["warmup_ratio"]) if config["training"]["warmup_ratio"] else 0

    print(f"Running with max_steps: {max_steps}, will start annealing at step: {initial_phase_steps}")
    run_name = f"v0_{config['models']['student'].split('/')[-1]}_lr_{config['training']['learning_rate']}"

    training_args = TrainingArguments(
        **config["training"],
        max_steps=max_steps, # Explicitly set max_steps
        num_train_epochs=config["training_aux"]["num_train_epochs"], # Set to None when using max_steps
        run_name=run_name,
        logging_dir=f"./logs/{run_name}",
        save_steps=save_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        # Default optimizer
        optim="adamw_torch",
        # Galore optimizer, uses 80% less memory than adamw_torch
        # optim="galore_adamw_8bit",
        # optim_target_modules=["mlp.down_proj", "mlp.gate_proj", "self_attn.q_proj", "self_att
        ddp_find_unused_parameters=False,
    )

    print(f"max_steps: {max_steps}")
    print(f"no. o f train epochs: {training_args.num_train_epochs}")

    if config.get("gradient_checkpointing", False):
        # Disable caching for gradient checkpointing compatibility
        trainer.model.config.use_cache = False

    # Prepare the trainer, models, and datasets
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset, # This is now a regular Dataset, not IterableDataset
        eval_dataset=eval_dataset,
        tokenizer=student_tokenizer,
        config=config, # This is your custom config, not SFTConfig
        dataset_text_field="text",
        max_seq_length=config["tokenizer"]["max_length"],
        # packing=False,
        packing=True,
    )

    if config.get("gradient_checkpointing", False)==True:
        # Disable caching for gradient checkpointing compatibility
        trainer.model.config.use_cache = False

    trainer, teacher_model, train_dataset, eval_dataset = accelerator.prepare(
        trainer, teacher_model, train_dataset, eval_dataset
    )

    trainer.teacher_model = teacher_model
    trainer.train_dataset = train_dataset
    trainer.eval_dataset = eval_dataset

    # Add custom scheduler
    optimizer = trainer.create_optimizer()
    scheduler = get_custom_lr_scheduler(optimizer, warmup_steps, max_steps, initial_phase_steps)
    trainer.lr_scheduler = scheduler

    trainer.add_callback(MemoryTracker())

    print("Starting knowledge distillation with evaluation...")
    try:
        trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    except RuntimeError as e:
        print("An error occurred during training: {e}")
        print("Please check that your GPU has enough memory and that all tensors are on the same device.")
        raise
    finally:
        print("Final memory stats:")
        print_memory_stats()

    print(f"Distillation completed. Saving model to {config['training']['output_dir']}")
    trainer.save_model(config['training']['output_dir'])

    trainer.push_to_hub()