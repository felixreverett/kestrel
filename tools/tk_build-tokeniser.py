# kestrel/tools/tk_build-tokeniser.py

import os
import argparse
import logging
import re
import torch
import json

from collections import Counter
from typing import List, Set, Tuple
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
#
# - [i] Currently hard-coded for Mauritian Creole
#       only. 
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
# - Helper function to load the datasets for the
#   KreolMorisienMT corpus.
#
# - [i] Tokenisation only requires monolingual data.
# ==================================================

def load_dataset(filepaths: List[str]) -> List[str]:
  corpus = []
  for path in filepaths:
    if os.path.exists(path):
      with open(path, "r", encoding="utf-8") as f:
        corpus.extend([line.strip() for line in f if line.strip()])
    else:
      logging.warning(f"[Kestrel] Could not find dataset file: {path}")
  return corpus

# ==================================================
# fn extract_words()
#
# - Extracts words from a given dataset.
# ==================================================

def extract_words(text: str) -> List[str]:
  return re.findall(r"\b\w+(?:-\w+)*\b", text, flags=re.UNICODE)

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
    #    That Ġ character conventionally represents whitespace :)
    clean_token = token.replace("Ġ", " ") if "Ġ" in token else token

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
# fn build_bpe_extended_tokeniser()
#
# - Finds the most frequent Mauritian Creole words
#   that the base Granite tokeniser splits into >= 2
#   subwords.
# ==================================================

def build_bpe_extended_tokeniser(base_tok: PreTrainedTokenizerFast, dataset: List[str], num_new_tokens: int, save_dir: str) -> Tuple[PreTrainedTokenizerFast, List[str]]:

  logging.info("[Kestrel] Counting word frequencies in Mauritian Creole dataset...")

  word_counts = Counter()
  sentence_initial_counts = Counter()

  for sentence in dataset:
    words = extract_words(sentence)
    if not words:
      continue
    sentence_initial_counts[words[0]] += 1
    word_counts.update(words)

  # Save base tokeniser to target directory
  os.makedirs(save_dir, exist_ok=True)
  base_tok.save_pretrained(save_dir)
  tok_json_path = os.path.join(save_dir, "tokenizer.json")

  with open(tok_json_path, "r", encoding="utf-8") as f:
    tok_data = json.load(f)

  vocab: dict = tok_data["model"]["vocab"]
  merges: list = tok_data["model"]["merges"]
  use_list_merges = len(merges) > 0 and isinstance(merges[0], list)

  existing_merges = set(
    (m[0], m[1]) if isinstance(m, list) else tuple(m.split(" "))
    for m in merges
  )

  # Build candidate strings (e.g. mid-sentence (" " + word) and common sentence-initial words)
  candidates = []
  for word, count in word_counts.most_common():
    if len(word) <= 1 or word.isnumeric():
      continue
    candidates.append((" " + word, count))
    if sentence_initial_counts[word] >= 5 and word[0].isupper():
      candidates.append((word, sentence_initial_counts[word]))

  # Sort candidates by frequency, descending
  candidates.sort(key=lambda x: x[1], reverse=True)

  added_tokens: List[str] = []
  next_id = max(vocab.values()) + 1

  for raw_surface, _ in candidates:
    if len(added_tokens) >= num_new_tokens:
      break

    # Check how base tokeniser currently segments this word
    sub_tokens = base_tok.tokenize(raw_surface)
    if len(sub_tokens) < 2:
      continue

    # Skip if any fragment isn't in the base BPE vocab
    if any(t not in vocab for t in sub_tokens):
      continue

    # Chain binary merges left-to-right
    curr_tokens = list(sub_tokens)
    while len(curr_tokens) > 1 and len(added_tokens) < num_new_tokens:
      left, right = curr_tokens[0], curr_tokens[1]
      merged_tok = left + right

      if (left, right) not in existing_merges:
        existing_merges.add((left, right))
        merges.append([left, right] if use_list_merges else f"{left} {right}")

      if merged_tok not in vocab:
        vocab[merged_tok] = next_id
        next_id += 1
        added_tokens.append(merged_tok)

      curr_tokens = [merged_tok] + curr_tokens[2:]

  # Write updated vocab and BPE merge rules back to tokenizer.json
  with open(tok_json_path, "w", encoding="utf-8") as f:
    json.dump(tok_data, f, ensure_ascii=False, indent=2)

  # Reload as clean PreTrainedTokenizerFast
  new_tok = AutoTokenizer.from_pretrained(save_dir)
  return new_tok, added_tokens

# ==================================================
# Main method
#
# - Boilerplate logging and command-line parsing.
# - 
# ==================================================

def main():
  logger = logging.getLogger(__name__)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(script_dir, ".."))

  # Parser
  parser = argparse.ArgumentParser(description="Build and initialise a custom CVA tokeniser for Mauritian Creole")
  parser.add_argument("-b", "--base_model", default="4.1-3b", help="Base model shortcut or HF ID")
  parser.add_argument("-n", "--num_tokens", type=int, default=1000, help="Number of new Creole tokens to add")
  parser.add_argument("-o", "--output_dir", default="granite-4.1-3b-creole-base", help="Output folder name in models/")
  args = parser.parse_args()

  # 1. Load dataset(s)
  # [i] currently hardcoded to use all three MorisienMT Creole datasets
  dataset_paths = [
    os.path.join(project_root, "datasets", "MorisienMT", "train.cr"),
    os.path.join(project_root, "datasets", "MorisienMT", "dev.cr"),
    os.path.join(project_root, "datasets", "train.en-cr.cr")
  ]
  dataset = load_dataset(dataset_paths)
  if not dataset:
    logger.error("[Kestrel] Dataset is empty.")
    return

  output_path = os.path.join(project_root, "models", args.output_dir)

  # 2. Load the base tokeniser
  base_path = resolve_tokeniser_path(args.base_model, project_root)
  logger.info(f"[Kestrel] Loading base tokeniser from {base_path}...")
  old_tok = AutoTokenizer.from_pretrained(base_path)

  # 3. Build BPE-extended tokeniser and inject new merge rules
  new_tok, new_tokens = build_bpe_extended_tokeniser(
    old_tok,
    dataset,
    num_new_tokens=args.num_tokens,
    save_dir=output_path
  )
  logger.info(f"[Kestrel] Injected {len(new_tokens):,} new BPE tokens and merge rules.")

  # 5. Load base model on CPU
  logger.info("[Kestrel] Loading base model on CPU to resize embeddings...")
  model = AutoModelForCausalLM.from_pretrained(
    base_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16
  )

  # 6. Resize and apply CVA Subword Mean Initialisation
  logger.info(f"[Kestrel] Resizing embeddings from {len(old_tok):,} to {len(new_tok):,}...")
  model.resize_token_embeddings(len(new_tok))

  logger.info("[Kestrel] Applying Subword Mean Initialisation (CVA)...")
  apply_subword_mean_initialisation(model, old_tok, new_tok, new_tokens)

  # 7. Export final model and tokeniser
  logger.info(f"[Kestrel] Saving CVA-adapted model and tokeniser to {output_path}...")
  model.save_pretrained(output_path)
  new_tok.save_pretrained(output_path)

if __name__ == "__main__":
  main()