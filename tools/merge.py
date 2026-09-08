# tools/merge.py

import os
import sys
import json
import logging
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
  logger = logging.getLogger(__name__)
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

  parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
  parser.add_argument("-m", "--model", choices=MODEL_REGISTRY.keys(), default=None, help="Base model shortcut. Auto-detected if omitted.")
  parser.add_argument("-a", "--adapter", required=True, help="Directory name of trained adapter")
  parser.add_argument("-o", "--output", default=None, help="Output directory for merged model")
  args = parser.parse_args()

  if args.output is None:
    args.output = f"{args.adapter}-final"

  # ==================================================
  # Resolve paths relative to tools/
  # ==================================================

  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(script_dir, ".."))
  
  adapter_dir = os.path.join(project_root, "adapters", args.adapter)
  output_dir = os.path.join(project_root, "models", args.output)

  if not os.path.exists(adapter_dir):
    logger.error(f"[Kestrel] ERROR: Adapter directory not found at {adapter_dir}")
    sys.exit(1)

  # ==================================================
  # Look for training-time config to validate model
  # ==================================================

  kestrel_config_path = os.path.join(adapter_dir, "kestrel_config.json")
  peft_config_path = os.path.join(adapter_dir, "adapter_config.json")

  detected_base_model_id = None

  if os.path.exists(kestrel_config_path):
    with open(kestrel_config_path, "r") as f:
      config_data = json.load(f)
      detected_base_model_id = config_data.get("base_model_id")
      logger.info(f"[Kestrel] Found kestrel_config.json. Trained base model: {detected_base_model_id}")

  elif os.path.exists(peft_config_path):
    with open(peft_config_path, "r") as f:
      config_data = json.load(f)
      detected_base_model_id = config_data.get("base_model_name_or_path")
      logger.info(f"[Kestrel] Found adapter_config.json. Trained base model: {detected_base_model_id}")

  else:
    logger.error(f"[Kestrel] ERROR: No config file found in {adapter_dir}. Cannot validate adapter weights.")
    sys.exit(1)

  # ==================================================
  # Strict validation guardrail
  # ==================================================
  
  if args.model:
    requested_base_model_id = MODEL_REGISTRY[args.model]
    if requested_base_model_id != detected_base_model_id:
      logger.critical(f"""\n[Kestrel] CRITICAL ERROR: Model mismatch!
        \n - Adapter was trained on : {detected_base_model_id}
        \n - You requested to merge : {requested_base_model_id}
        \nAborting process to prevent corrupted weights.\n""")
      sys.exit(1)
    base_model_id = requested_base_model_id

  else:
    if detected_base_model_id not in MODEL_REGISTRY.values():
      logger.error(f"[Kestrel] ERROR: Detected model '{detected_base_model_id}' not in MODEL_REGISTRY.")
      sys.exit(1)
    base_model_id = detected_base_model_id
    logger.info(f"[Kestrel] Auto-detected base model ID: {base_model_id}")
  
  # ==================================================
  # Load base model
  # ==================================================

  logger.info(f"[Kestrel] Loading original base model: {base_model_id}")
  base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
  )

  tokenizer = AutoTokenizer.from_pretrained(base_model_id)
  
  logger.info(f"[Kestrel] Loading trained adapters from {adapter_dir}...")
  model = PeftModel.from_pretrained(base_model, adapter_dir)

  logger.info("[Kestrel] Merging weights...")
  model = model.merge_and_unload()

  logger.info(f"[Kestrel] Saving merged model to {output_dir}...")
  model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  logger.info("[Kestrel] Merge complete! Model ready for GGUF conversion.")

if __name__ == "__main__":
  main()