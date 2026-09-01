# Training script to fine-tune IBM Granite model

import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

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

  # ==================================================
  # Set up command-line flag interpreter via argparse
  # ==================================================

  parser = argparse.ArgumentParser(description="Fine-tune IBM Granite")
  parser.add_argument("-m", "--model", choices=MODEL_REGISTRY.keys(), default="4.1-3b", help="Select model")
  parser.add_argument("-d", "--dataset", default="English-Mauritian Creole-bidirectional.jsonl", help="Dataset filename from datasets folder")
  parser.add_argument("-o", "--output", default=None, help="Output directory name for adapter")
  parser.add_argument("-bs", "--batchsize", default=2, help="Batch size for training")
  parser.add_argument("-ms", "--maxsteps", default=-1, help="Max steps for training. Default -1 to ignore")
  args = parser.parse_args()

  if args.output is None:
    args.output = f"adapters/kestrel-{args.model}"

  # ==================================================
  # Set variables based on flags
  # ==================================================

  model_id = MODEL_REGISTRY[args.model]

  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(script_dir, ".."))
  
  dataset_path = os.path.join(project_root, "datasets", args.dataset)
  output_dir = os.path.join(project_root, args.output)

  print(f"[Kestrel] Using model: {model_id}")
  print(f"[Kestrel] Loading tokeniser...")

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

  print("[Kestrel] Loading and formatting dataset...")
  dataset = load_dataset("json", data_files=dataset_path, split="train")

  def format_chat(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, truncation=True, max_length=512)

  dataset = dataset.map(format_chat, batched=False, remove_columns=dataset.column_names)

  # ==================================================
  # Set up bits and bytes configuration
  # ==================================================

  print("[Kestrel] Configuring 4-bit quantisation and loading model...")
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
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

  print("[Kestrel] Injecting LoRA adapters...")

  lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
  )

  batch_size = int(args.batchsize)
  grad_accum = 8 if batch_size == 1 else 4

  training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=args.maxsteps,
    optim="paged_adamw_8bit",
    bf16=True,
    loss_type="nll",
    router_aux_loss_coef=0.0,
    save_strategy="steps",
    save_steps=100,
    gradient_checkpointing_kwargs={"use_reentrant": False},
  )

  # ==================================================
  # Initialise trainer
  # ==================================================

  print("[Kestrel] Initialising trainer...")

  trainer = SFTTrainer(
    model=model,# type: ignore
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
  )

  # ==================================================
  # Train (takes a while, to put it mildly)
  # ==================================================

  print("[Kestrel] Starting training...")

  # Check if a checkpoint exists to resume automatically
  last_checkpoint = None
  if os.path.isdir(output_dir):
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
      last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
      print(f"[Kestrel] Found existing checkpoint! Resuming from: {last_checkpoint}")

  trainer.train(resume_from_checkpoint=last_checkpoint)

  print(f"[Kestrel] Saving final adapter to {output_dir}...")
  if trainer.model is not None:
    trainer.model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  # ==================================================
  # Output config for merge stage
  # ==================================================

  config_path = os.path.join(output_dir, "kestrel_config.json")
  with open(config_path, "w") as f:
    json.dump({"model_shortcut": args.model, "base_model_id": model_id}, f, indent=2)
  print(f"[Kestrel] Saved training config to {config_path}")

  print("[Kestrel] Test run complete. Ending program.")

if __name__ == "__main__":
  main()