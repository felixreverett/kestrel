# tools/merge.py

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================================================
# List of possible models. To add a model option,
# simply add its URL below.
# ==================================================

MODEL_REGISTRY = {
  "4.1-3b": "ibm-granite/granite-4.1-3b-base",
  "4.1-8b": "ibm-granite/granite-4.1-8b-base",
  "3.0-3b": "ibm-granite/granite-3.0-3b-a800m-base"
}

def main():
  parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
  parser.add_argument("-m", "--model", choices=MODEL_REGISTRY.keys(), default=None, help="Base model shortcut. If omitted, auto-detected from the adapter's kestrel_config.json")
  parser.add_argument("-a", "--adapter", default="kestrel-4.1-8b", help="Directory name of trained adapter")
  parser.add_argument("-o", "--output", default="kestrel-4.1-8b-final", help="Output directory for merged model")
  args = parser.parse_args()

  base_model_id = MODEL_REGISTRY[args.model]

  # ==================================================
  # Resolve paths relative to tools/
  # ==================================================

  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(script_dir, ".."))
  
  adapter_dir = os.path.join(project_root, "adapters", args.adapter)
  output_dir = os.path.join(project_root, "models", args.output)

  # ==================================================
  # Look for training-time config saved by train.py
  # ==================================================

  config_path = os.path.join(adapter_dir, "kestrel_config.json")
  saved_shortcut = None

  if os.path.exists(config_path):
    with open(config_path) as f:
      saved_shortcut = json.load(f).get("model_shortcut")

  if args.model is None:
    if saved_shortcut is not None:
      args.model = saved_shortcut
      print(f"[Kestrel] Auto-detected base model from adapter config: {args.model}")
    else:
      args.model = "4.1-8b"
      print(f"[Kestrel] No kestrel_config.json found in adapter dir, defaulting to {args.model}")
  elif saved_shortcut is not None and saved_shortcut != args.model:
    print(f"[Kestrel] WARNING: adapter was trained with '{saved_shortcut}' but model '-m {args.model} was specified'. Proceeding with '{args.model}'; this will likely produce a broken merge.")

  # ==================================================
  # Load base model
  # ==================================================

  print(f"[Kestrel] Loading original base model: {base_model_id}")
  base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
  )

  tokenizer = AutoTokenizer.from_pretrained(base_model_id)
  
  print(f"[Kestrel] Loading trained adapters from {adapter_dir}...")
  model = PeftModel.from_pretrained(base_model, adapter_dir)

  print("[Kestrel] Merging weights...")
  model = model.merge_and_unload()

  print(f"[Kestrel] Saving merged model to {output_dir}...")
  model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  print("[Kestrel] Merge complete! Model ready for GGUF conversion.")

if __name__ == "__main__":
  main()