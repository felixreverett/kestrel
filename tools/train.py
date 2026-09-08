# Training script to fine-tune IBM Granite model

import os
import argparse
import json
import torch
import logging
from dataclasses import dataclass, field, asdict
from datasets import load_dataset
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from typing import Literal


# ==================================================
# Configs
# ==================================================

@dataclass
class BnbConfig:
  load_in_4bit: bool = True
  bnb_4bit_use_double_quant: bool = True
  bnb_4bit_quant_type: str = "nf4"
  bnb_4bit_compute_dtype: str = "bfloat16"

@dataclass
class LoraHyperConfig:
  r: int = 16
  lora_alpha: int = 32
  target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
  lora_dropout: float = 0.05
  bias: Literal["none", "all", "lora_only"] = "none"
  task_type: str = "CAUSAL_LM"

@dataclass
class TrainingHyperConfig:
  per_device_train_batch_size: int = 2
  gradient_accumulation_steps: int = 4
  learning_rate: float = 2e-4
  logging_steps: int = 10
  max_steps: int = -1
  optim: str = "paged_adamw_8bit"
  bf16: bool = True
  loss_type: str = "nll"
  save_strategy: str = "steps"
  save_steps: int = 100
  max_seq_length: int = 512

@dataclass
class KestrelConfig:
  model: str = "4.1-3b"
  dataset: str = "English-Mauritian Creole-bidirectional.jsonl"
  output_dir: str = ""
  bnb: BnbConfig = field(default_factory=BnbConfig)
  lora: LoraHyperConfig = field(default_factory=LoraHyperConfig)
  training: TrainingHyperConfig = field(default_factory=TrainingHyperConfig)

  @classmethod
  def from_json(cls, json_path: str):
    """Hydrates the nested struct objects safely from a JSON file."""
    with open(json_path, "r") as f:
      data = json.load(f)

    if "bnb" in data:
      data["bnb"] = BnbConfig(**data["bnb"])
    if "lora" in data:
      data["lora"] = LoraHyperConfig(**data["lora"])
    if "training" in data:
      data["training"] = TrainingHyperConfig(**data["training"])

    return cls(**data)


# ==================================================
# List of possible models. To add a model option,
# simply add its URL below.
# ==================================================

MODEL_REGISTRY= {
  "4.1-3b": "ibm-granite/granite-4.1-3b-base",
  "4.1-8b": "ibm-granite/granite-4.1-8b-base",
  "3.0-3b": "ibm-granite/granite-3.0-3b-a800m-base"
}


# ==================================================
# Main method:
# - Sets up command-line flag parsing
# - Sets up tokeniser
# - Loads dataset
# - Sets up bits and bytes config
# - Injects LoRA adapters
# - Initialises trainer
# - Trains adapter weights
# - Exports adapter to adapters/
# ==================================================

def main():

  logger = logging.getLogger(__name__)
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(script_dir, ".."))

  # ==================================================
  # Set up command-line flag interpreter via argparse
  # ==================================================

  parser = argparse.ArgumentParser(description="Fine-tune IBM Granite")
  parser.add_argument("-c", "--config", default=None, help="Path to JSON config. Overrides all other flags if provided")
  parser.add_argument("-m", "--model", choices=MODEL_REGISTRY.keys(), default="4.1-3b", help="Select model")
  parser.add_argument("-d", "--dataset", default="English-Mauritian Creole-bidirectional.jsonl", help="Dataset filename from datasets folder")
  parser.add_argument("-o", "--output", default=None, help="Output directory name for adapter")
  parser.add_argument("-bs", "--batchsize", default=2, help="Batch size for training")
  parser.add_argument("-ms", "--maxsteps", default=-1, help="Max steps for training. Default -1 to ignore")
  args = parser.parse_args()

  # ==================================================
  # Build configs
  # ==================================================

  cfg = None
  if args.config:
    if os.path.exists(args.config):
      resolved_config_path = args.config
    else:
      resolved_config_path = os.path.join(project_root, "configs", args.config)

    if os.path.exists(resolved_config_path):
      logger.info(f"[Kestrel] Loading hyperparameters from config file: {resolved_config_path}")
      cfg = KestrelConfig.from_json(resolved_config_path)
    else:
      logger.warning(f"[Kestrel] WARNING: Config file {resolved_config_path} not found. Using CLI defaults.")

  if cfg is None:
    cfg = KestrelConfig(
      model=args.model,
      dataset=args.dataset,
      output_dir=args.output or f"adapters/kestrel-{args.model}"
    )
    cfg.training.per_device_train_batch_size = int(args.batchsize)
    cfg.training.gradient_accumulation_steps = 8 if int(args.batchsize) == 1 else 4
    cfg.training.max_steps = int(args.maxsteps)

  # ==================================================
  # Set variables based on flags
  # ==================================================

  model_id = MODEL_REGISTRY[cfg.model]

  dataset_path = os.path.join(project_root, "datasets", cfg.dataset)
  output_dir = os.path.join(project_root, cfg.output_dir or f"adapters/kestrel-{cfg.model}")

  logger.info(f"[Kestrel] Using model: {model_id}")
  logger.info(f"[Kestrel] Loading tokeniser...")

  # ==================================================
  # Initialise tokeniser
  # ==================================================

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}"

  # ==================================================
  # Load dataset
  # ==================================================

  logger.info("[Kestrel] Loading and formatting dataset...")
  dataset = load_dataset("json", data_files=dataset_path, split="train")

  def format_chat(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, truncation=True, max_length=cfg.training.max_seq_length)

  dataset = dataset.map(format_chat, batched=False, remove_columns=dataset.column_names)

  # ==================================================
  # Set up bits and bytes configuration
  # ==================================================

  logger.info("[Kestrel] Configuring 4-bit quantisation and loading model...")

  compute_dtype = torch.bfloat16 if cfg.bnb.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

  bnb_config = BitsAndBytesConfig(
    load_in_4bit=cfg.bnb.load_in_4bit,
    bnb_4bit_use_double_quant=cfg.bnb.bnb_4bit_use_double_quant,
    bnb_4bit_quant_type=cfg.bnb.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
  )

  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0",
    use_cache=False,
    low_cpu_mem_usage=True,
  )

  model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

  # ==================================================
  # Inject LoRA adapters
  # ==================================================

  logger.info("[Kestrel] Injecting LoRA adapters...")

  lora_config = LoraConfig(
    r=cfg.lora.r,
    lora_alpha=cfg.lora.lora_alpha,
    target_modules=cfg.lora.target_modules,
    lora_dropout=cfg.lora.lora_dropout,
    bias=cfg.lora.bias,
    task_type=cfg.lora.task_type
  )

  training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=cfg.training.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    learning_rate=cfg.training.learning_rate,
    logging_steps=cfg.training.logging_steps,
    max_steps=cfg.training.max_steps,
    optim=cfg.training.optim,
    bf16=cfg.training.bf16,
    loss_type=cfg.training.loss_type,
    save_strategy=cfg.training.save_strategy,
    save_steps=cfg.training.save_steps,
    max_length=cfg.training.max_seq_length,
    gradient_checkpointing_kwargs={"use_reentrant": False},
  )

  # ==================================================
  # Initialise trainer
  # ==================================================

  logger.info("[Kestrel] Initialising trainer...")

  trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
  )

  # ==================================================
  # Train (takes a while, to put it mildly)
  # ==================================================

  logger.info("[Kestrel] Starting training...")

  # Check if a checkpoint exists to resume automatically
  last_checkpoint = None
  if os.path.isdir(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
      last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
      logger.info(f"[Kestrel] Found existing checkpoint! Resuming from: {last_checkpoint}")

  trainer.train(resume_from_checkpoint=last_checkpoint)

  logger.info(f"[Kestrel] Saving final adapter to {output_dir}...")
  if trainer.model is not None:
    trainer.model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  # ==================================================
  # Output config for merge stage
  # ==================================================

  config_path = os.path.join(output_dir, "kestrel_config.json")
  with open(config_path, "w") as f:
    json.dump({
        "model_shortcut": cfg.model, 
        "base_model_id": model_id,
        "hyperparameters": asdict(cfg)
    }, f, indent=2)
  
  logger.info(f"[Kestrel] Saved training config to {config_path}")

  logger.info("[Kestrel] Training complete. Ending program.")

if __name__ == "__main__":
  main()