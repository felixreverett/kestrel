# kestrel/tools/tk_build-tokeniser.py

import os
import argparse
import logging
import torch

from typing import List, Set
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  PreTrainedTokenizerFast,
  PreTrainedModel
)

# ==================================================
# tk_build-tokeniser.py
#
# - Trains a custom Mauritian Creole tokeniser by
#   extending the base Granite 4.1 vocabulary.
# - Applies Subword Mean Initialisation (CVA) to
#   prevent catastrophic forgetting on smaller
#   datasets.
# ==================================================

MODEL_REGISTRY = {
  "4.1-3b": "ibm-granite/granite-4.1-3b-base",
  "4.1-8b": "ibm-granite/granite-4.1-8b-base",
  "3.0-3b": "ibm-granite/granite-3.0-3b-a800m-base",
}

# ==================================================
# fn resolve_tokeniser_path()
#
# - Resolves model shortcut, local dir, or HF ID.
# ==================================================

def resolve_tokeniser_path(path: str, root: str) -> str:
  if path in MODEL_REGISTRY:
    return MODEL_REGISTRY[path]
  for subfolder in ["", "models", "adapters"]:
    candidate = os.path.join(root, subfolder, path) if subfolder else os.path.join(root, path)
    if os.path.exists(candidate):
      return candidate
  return path

# ==================================================
# fn load_creole_corpus()
#
# - Loads the Mauritian Creole text from the KMMT
#   dataset.
# ==================================================

def load_creole_corpus(filepaths: List[str]) -> List[str]:
  corpus = []
  for path in filepaths:
    if os.path.exists(path):
      with open(path, "r", encoding="utf-8") as f:
        corpus.extend([line.strip() for line in f if line.strip()])
    else:
      logging.warning(f"[Kestrel] Could not find corpus file: {path}")
  return corpus

# ==================================================
# fn extract_new_tokens()
#
# - Trains a new Byte-Pair Encoding (BPE) tokeniser
#   on the Creole corpus.
# - Filters out tokens that already exist in the
#   base Granite vocabulary.
# ==================================================

def extract_new_tokens(base_tok: PreTrainedTokenizerFast, corpus: List[str], target_vocab_size: int, num_new_tokens: int) -> List[str]:

  # Use a generator to feed text to the tokeniser trianer
  def batch_iterator(batch_size=1000):
    for i in range(0, len(corpus), batch_size):
      yield corpus[i : i + batch_size]

  logging.info("[Kestrel] Training temporary BPE tokeniser on Mauritian Creole corpus...")

  # Train new tokeniser from scratch on KMMT corpus
  temp_tok = base_tok.train_new_from_iterator(batch_iterator(), vocab_size=target_vocab_size)

  # Extract vocabulary dictionary (token -> ID)
  temp_vocab = temp_tok.get_vocab()
  base_vocab = base_tok.get_vocab()

  sorted_temp_tokens = sorted(temp_vocab.keys(), key=lambda k: temp_vocab[k])

  new_tokens = []

  for token in sorted_temp_tokens:
    if token not in base_vocab:
      # Ignore single-char tokens or byte fallbacks
      if len(token) > 1 and not token.startswith("<0x"):
        new_tokens.append(token)
    if len(new_tokens) >= num_new_tokens:
      break

  return new_tokens

# ==================================================
# fn apply_subword_mean_initialisation()
#
# - Initialises the new KM embeddings by averaging
#   the Granite subwords they would have otherwise
#   been split into (this is Cross-lingual
#   Vocabulary Adaptation).
# ==================================================

def apply_subword_mean_initialisation(
  model: PreTrainedModel,
  old_tok: PreTrainedTokenizerFast,
  new_tok: PreTrainedTokenizerFast,
  added_tokens: List[str]):

  in_module = model.get_input_embeddings()
  out_module = model.get_output_embeddings()

  assert in_module is not None, "Model lacks input embeddings."
  assert out_module is not None, "Model lacks output embeddings."

  input_embeddings = in_module.weight.data
  output_embeddings = out_module.weight.data

  for token in added_tokens:
    # 1. Get ID of new token in new tokeniser
    new_id = new_tok.convert_tokens_to_ids(token)
    # 2. Tokenise same string using old tokeniser
    clean_token = token.replace(" ", " ") if " " in token else token
    old_ids = old_tok.encode(clean_token, add_special_tokens=False)

    if not old_ids:
      continue

    # 3. Fetch old embedding vectors
    old_in_embeds = input_embeddings[old_ids]
    old_out_embeds = output_embeddings[old_ids]

    # 4. Compute mean across fragments
    mean_in_embed = old_in_embeds.mean(dim=0)
    mean_out_embed = old_out_embeds.mean(dim=0)

    # 5. Assign the averaged vector to the new token's index
    input_embeddings[new_id] = mean_in_embed
    output_embeddings[new_id] = mean_out_embed

# ==================================================
# Main method
# ==================================================

def main():
  logger = logging.getLogger(__name__)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(script_dir, ".."))

  parser = argparse.ArgumentParser(description="Build and initialise a custom CVA tokeniser for Mauritian Creole")
  parser.add_argument("-b", "--base_model", default="4.1-3b", help="Base model shortcut or HF ID")
  parser.add_argument("-n", "--num_tokens", type=int, default=1500, help="Number of new tokens to add")
  parser.add_argument("-o", "--output_dir", default="granite-4.1-3b-tk", help="Output folder name in models/")
  args = parser.parse_args()

  # 1. Load dataset (train and dev sets from KreolMorisienMT)
  corpus_paths = [
    os.path.join(project_root, "datasets", "MorisienMT", "train.cr"),
    os.path.join(project_root, "datasets", "MorisienMT", "dev.cr")
  ]
  corpus = load_creole_corpus(corpus_paths)
  if not corpus:
    logger.error("[Kestrel] Corpus is empty. Please ensure MorisienMT/train.cr exists.")
    return

  logger.info(f"[Kestrel] Loaded {len(corpus):,} Mauritian Creole sentences.")

  # 2. Load base tokeniser
  base_path = resolve_tokeniser_path(args.base_model, project_root)
  logger.info(f"[Kestrel] Loading base tokeniser from {base_path}...")
  old_tok = AutoTokenizer.from_pretrained(base_path)
  
  # 3. Extract new tokens
  new_tokens = extract_new_tokens(old_tok, corpus, target_vocab_size=10000, num_new_tokens=args.num_tokens)
  logger.info(f"[Kestrel] Identified {len(new_tokens):,} new Mauritian Creole tokens.")
  
  # 4. Create new tokeniser and add tokens
  new_tok = AutoTokenizer.from_pretrained(base_path)
  new_tok.add_tokens(new_tokens)
  
  # 5. Load the base model (on CPU)
  logger.info(f"[Kestrel] Loading base model on CPU to resize embeddings...")
  model = AutoModelForCausalLM.from_pretrained(
    base_path, 
    device_map="cpu", 
    torch_dtype=torch.bfloat16
  )
  
  # 6. Resize embeddings and apply CVA
  logger.info(f"[Kestrel] Resizing embeddings from {len(old_tok):,} to {len(new_tok):,}...")
  model.resize_token_embeddings(len(new_tok))
  
  logger.info("[Kestrel] Applying Subword Mean Initialisation (CVA)...")
  apply_subword_mean_initialisation(model, old_tok, new_tok, new_tokens)
  
  # 7. Export new base model and tokeniser
  output_path = os.path.join(project_root, "models", args.output_dir)
  os.makedirs(output_path, exist_ok=True)
  
  logger.info(f"[Kestrel] Saving CVA-adapted model and tokeniser to {output_path}...")
  model.save_pretrained(output_path)
  new_tok.save_pretrained(output_path)
  
  logger.info("\n[Kestrel] Success! You can now fine-tune this model.")
  logger.info(f"Update train.py to point to: models/{args.output_dir}")

if __name__ == "__main__":
  main()