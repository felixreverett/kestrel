# Training script to fine-tune IBM Granite model

import torch
from datasets import load_dataset
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

# Config
model_id = "ibm-granite/granite-3.0-3b-a800m-base"
dataset_path = "datasets/English-Mauritian Creole-bidirectional.jsonl"
output_dir = "./kestrel-3b-adapter"

print("[Kestrel] Loading tokeniser...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}"

print("[Kestrel] Loading and formatting dataset...")
dataset = load_dataset("json", data_files = dataset_path, split="train")

def format_chat(example):
  text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
  return tokenizer(text, truncation=True, max_length=512)

dataset = dataset.map(format_chat, batched=False, remove_columns=dataset.column_names)

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
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

print("[Kestrel] Injecting LoRA adapters...")

lora_config = LoraConfig(
  r=16,
  lora_alpha=32,
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
  lora_dropout=0.05,
  bias="none",
  task_type="CAUSAL_LM"
)

training_args = SFTConfig(
  output_dir=output_dir,
  per_device_train_batch_size=2,
  gradient_accumulation_steps=4,
  learning_rate=2e-4,
  logging_steps=10,
  max_steps=100,
  optim="paged_adamw_8bit",
  bf16=True,
  loss_type="nll",
  router_aux_loss_coef=0.0,
  save_strategy="steps",
  save_steps=50,
)

print("[Kestrel] Initialising trainer...")

trainer = SFTTrainer(
  model=model,# type: ignore
  train_dataset=dataset,
  peft_config=lora_config,
  processing_class=tokenizer,
  args=training_args,
)

print("[Kestrel] Starting training...")
trainer.train()

print(f"[Kestrel] Saving final adapter to {output_dir}...")
if trainer.model is not None:
  trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[Kestrel] Test run complete")