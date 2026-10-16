# kestrel/tools/tk_fertility-eval.py

import os
import re
import json
import argparse
import logging

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer

# ==================================================
# tk_fertility-eval.py
#
# - Evaluates the token fertility rate of a model,
#   where a fertility rate around 1.0 is optimal.
# - This produces an objective metric for a model's
#   ability to effectively tokenise a language,
#   which might strengthen Kestrel models
# ==================================================

MODEL_REGISTRY = {
  "4.1-3b": "ibm-granite/granite-4.1-3b-base",
  "4.1-8b": "ibm-granite/granite-4.1-8b-base",
  "3.0-3b": "ibm-granite/granite-3.0-3b-a800m-base",
}

# ==================================================
# Sample Mauritian Creole sentences from the thesis
# for qualitative linguistic and orthographic checks
# ==================================================

DIAGNOSTIC_SAMPLES = [
  "en simíz rúz",
  "en simíz ruz-rúz",
  'Enn gro lavwa dir: "Kisannla pe manz fey dan mo zardin?".',
  "Enn lavil ki finn konstrir lor enn montagn pa kapav kasiet.",
  "Enn zarb bon pa kapav prodwir move fri, e enn zarb pa bon pa kapav prodwir bon fri.",
  "Tipikman, flas la ti dir enn fraksion enn segonn apre kolision.",
  "Kan li arive dan fon dan lamer, Pekoy truv buku pwason.",
  "Kan li ariv enn bagay lapas profond lor lamer, Pekoy trouv boukou pwason.",
]

# ==================================================
# Dataclass LanguageMetadata
#
# - Stores intrinsic tokenisation and fertility
#   metrics for a single language corpus.
# ==================================================

@dataclass
class LanguageMetadata:
  language: str
  num_sentences: int
  total_words: int
  total_chars: int
  total_seq_tokens: int
  total_word_tokens: int
  subword_fertility: float # Mean tokens per word
  seq_fertility: float # Mean tokens per word (sequence-level)
  continued_word_rate: float
  chars_per_token: float
  unknown_token_count: int

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
# fn infer_output_filename()
# ==================================================

def infer_output_filename(tokeniser_path_or_shortcut: str) -> str:
  text = tokeniser_path_or_shortcut.lower()
  
  # Determine model type (base vs kestrel)
  is_kestrel = "kestrel" in text or "adapter" in text or "models/" in text
  model_type = "kestrel" if is_kestrel else "base"
  
  # Determine size (3b vs 8b)
  if "8b" in text:
    size = "4.1-8b"
  elif "3.0-3b" in text:
    size = "3.0-3b"
  else:
    size = "4.1-3b"
    
  # Determine language variant (fr vs en)
  lang = "fr" if "fr" in text else "en"
  
  return f"tokeniser_{model_type}-{size}-{lang}.json"

# ==================================================
# fn extract_words()
# 
# - Splits text into words and strips punctuation.
# ==================================================

def extract_words(text: str) -> List[str]:
  return re.findall(r"\b\w+(?:-\w+)*\b", text, flags=re.UNICODE)

# ==================================================
# fn load_kestrel_jsonl()
#
# - Parses a ChatML .jsonl file.
# - Deduplicates sentences per language.
# - Extracts aligned pairs.
# ==================================================

def load_kestrel_jsonl(path: str) -> Tuple[Dict[str, List[str]], List[Tuple[str, str, str, str]]]:
  language_sets: Dict[str, Dict[str, None]] = {}
  aligned_pairs: List[Tuple[str, str, str, str]] = []
  seen_pairs = set()

  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      item = json.loads(line)
      messages = item.get("messages", [])

      system_text, user_text, target_text = "", "", ""
      for m in messages:
        if m["role"] == "system":
          system_text = m["content"]
        elif m["role"] == "user":
          user_text = m["content"]
        elif m["role"] == "assistant":
          target_text = m["content"]

      # This bit is hardcoded for the exact prompts used in Kestrel.
      # Changes in prompt will require a change here.
      match_en = re.search(r"Translate the following (.*?) text into (.*?)\.?$", system_text.strip())
      match_fr = re.search(r"Traduisez le texte (.*?) suivant en (.*?)\.?$", system_text.strip())

      if match_en:
        src_lang, tgt_lang = match_en.group(1).strip(), match_en.group(2).strip()
      elif match_fr:
        src_lang, tgt_lang = match_fr.group(1).strip(), match_fr.group(2).strip()
      else:
        src_lang, tgt_lang = "Source", "Target"

      language_sets.setdefault(src_lang, {})[user_text] = None
      language_sets.setdefault(tgt_lang, {})[target_text] = None

      pair_key = tuple(sorted([(src_lang, user_text), (tgt_lang, target_text)]))
      if pair_key not in seen_pairs:
        seen_pairs.add(pair_key)
        aligned_pairs.append((src_lang, user_text, tgt_lang, target_text))

  lang_sentences = {lang: list(sentences.keys()) for lang, sentences in language_sets.items()}
  return lang_sentences, aligned_pairs

# ==================================================
# fn load_raw_parallel()
#
# - Loads line-by-line parallel files.
# ==================================================

def load_raw_parallel(src_path: str, src_lang: str, tgt_path: str, tgt_lang: str) -> Tuple[Dict[str, List[str]], List[Tuple[str, str, str, str]]]:
  with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
    src_lines = [line.strip() for line in f_src if line.strip()]
    tgt_lines = [line.strip() for line in f_tgt if line.strip()]

  min_len = min(len(src_lines), len(tgt_lines))
  src_lines, tgt_lines = src_lines[:min_len], tgt_lines[:min_len]

  language_sentences = {src_lang: src_lines, tgt_lang: tgt_lines}
  aligned_pairs = [(src_lang, s, tgt_lang, t) for s, t in zip(src_lines, tgt_lines)]

  return language_sentences, aligned_pairs

# ==================================================
# fn evaluate_language_corpus
#
# - Takes a tokeniser, a language, and a
#   list of sentences.
# - Produces a LanguageMetadata object containing
#   the fertility of that tokeniser on the given data.
# ==================================================

def evaluate_language_corpus(tokeniser: Any, language: str, sentences: List[str]) -> LanguageMetadata:
  total_words = 0
  total_chars = 0
  total_seq_tokens = 0
  total_word_tokens = 0
  continued_words = 0
  unk_count = 0
  unk_id = tokeniser.unk_token_id

  # Iterate over all sentences to count sequence-level and word-level tokens
  for sentence in sentences:
    total_chars += len(sentence)
    seq_ids = tokeniser.encode(sentence, add_special_tokens=False)
    total_seq_tokens += len(seq_ids)
    if unk_id is not None:
      unk_count += sum(1 for tid in seq_ids if tid == unk_id)

    words = extract_words(sentence)
    total_words += len(words)

    for word in words:
      word_ids = tokeniser.encode(" " + word, add_special_tokens=False)
      if not word_ids:
        word_ids = tokeniser.encode(word, add_special_tokens=False)
      num_subwords = len(word_ids)
      total_word_tokens += num_subwords
      if num_subwords >= 2:
        continued_words += 1

  # Calculate fertility and compression rates
  subword_fertility = total_word_tokens / max(1, total_words)
  seq_fertility = total_seq_tokens / max(1, total_words)
  continued_word_rate = (continued_words / max(1, total_words)) * 100.0
  chars_per_token = total_chars / max(1, total_seq_tokens)

  # Return populated LanguageMetadata dataclass
  return LanguageMetadata(
    language=language,
    num_sentences=len(sentences),
    total_words=total_words,
    total_chars=total_chars,
    total_seq_tokens=total_seq_tokens,
    total_word_tokens=total_word_tokens,
    subword_fertility=round(subword_fertility, 4),
    seq_fertility=round(seq_fertility, 4),
    continued_word_rate=round(continued_word_rate, 2),
    chars_per_token=round(chars_per_token, 4),
    unknown_token_count=unk_count,
  )

# ==================================================
# fn compute_parity_ratio()
#
# - Calculates the cross-lingual parity ratio:
#   mean Token(Mauritian Creole) / Token(English/French)
#   across aligned sentence pairs.
# - A ratio of 1.0 indicates equal token compression
#   between Mauritian Creole and the high-resource
#   language.
# ==================================================

def compute_parity_ratio(tokenizer: Any, aligned_pairs: List[Tuple[str, str, str, str]], creole_name: str = "Mauritian Creole") -> Optional[float]:
  if not aligned_pairs:
    return None

  creole_tokens_total = 0
  other_tokens_total = 0

  for src_lang, src_text, tgt_lang, tgt_text in aligned_pairs:
    if src_lang.lower() == creole_name.lower():
      cr_text, ref_text = src_text, tgt_text
    elif tgt_lang.lower() == creole_name.lower():
      cr_text, ref_text = tgt_text, src_text
    else:
      continue

    creole_tokens_total += len(tokenizer.encode(cr_text, add_special_tokens=False))
    other_tokens_total += len(tokenizer.encode(ref_text, add_special_tokens=False))

  if other_tokens_total == 0:
    return None
  return round(creole_tokens_total / other_tokens_total, 4)

# ==================================================
# fn run_diagnostic_samples()
#
# - Tokenises the Mauritian Creole diagnostic
#   sentences for qualitative inspection of
#   morphemes, reduplication, and spelling variants.
# ==================================================

def run_diagnostic_samples(tokenizer: Any) -> List[Dict[str, Any]]:
  diagnostics = []
  for sample in DIAGNOSTIC_SAMPLES:
    tokens = tokenizer.tokenize(sample)
    words = extract_words(sample)
    diagnostics.append({
    "sentence": sample,
    "word_count": len(words),
    "token_count": len(tokens),
    "sample_fertility": round(len(tokens) / max(1, len(words)), 2),
    "tokens": tokens,
    })
  return diagnostics

# ==================================================
# Main method:
# - Sets up command-line flag parsing
# - Loads JSONL or raw parallel dataset
# - Evaluates base tokeniser fertility
# - Optionally compares against a custom tokeniser
# - Prints qualitative segmentation diagnostics
# - Exports results to evaluation/
# ==================================================

def main():
  logger = logging.getLogger(__name__)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.abspath(os.path.join(script_dir, ".."))

  # ==================================================
  # Set up command-line flag interpreter via argparse
  # ==================================================

  parser = argparse.ArgumentParser(description="Evaluate tokeniser fertility and efficiency for Kestrel")
  parser.add_argument("-b", "--base_tokeniser", "--base_tokenizer", dest="base_tokeniser", default="4.1-3b", help="Base model shortcut (4.1-3b, 4.1-8b), HF ID, or local path")
  parser.add_argument("-c", "--custom_tokeniser", "--custom_tokenizer", dest="custom_tokeniser", default=None, help="Optional custom tokeniser path to compare against the base tokeniser")
  parser.add_argument("-d", "--dataset", default="English-Mauritian Creole-test.jsonl", help="JSONL dataset filename in datasets/ folder")
  parser.add_argument("--raw_cr", default=None, help="Optional raw Mauritian Creole file in datasets/ (e.g. MorisienMT/dev.cr)")
  parser.add_argument("--raw_other", default=None, help="Optional raw aligned English/French file in datasets/ (e.g. MorisienMT/dev.en)")
  parser.add_argument("--other_lang", default="English", help="Language name when using --raw_other")
  parser.add_argument("-o", "--output_file", default=None, help="Optional explicit output filename override in evaluation/ subfolder")
  args = parser.parse_args()

  # ==================================================
  # Load dataset
  # ==================================================

  if args.raw_cr and args.raw_other:
    cr_path = os.path.join(project_root, "datasets", args.raw_cr)
    other_path = os.path.join(project_root, "datasets", args.raw_other)
    logger.info(f"[Kestrel] Loading raw parallel files: {cr_path} & {other_path}")
    lang_sentences, aligned_pairs = load_raw_parallel(cr_path, "Mauritian Creole", other_path, args.other_lang)
  else:
    dataset_path = os.path.join(project_root, "datasets", args.dataset)
    logger.info(f"[Kestrel] Loading JSONL dataset: {dataset_path}")
    lang_sentences, aligned_pairs = load_kestrel_jsonl(dataset_path)

  # ==================================================
  # Load and evaluate base tokeniser
  # ==================================================

  base_path = resolve_tokeniser_path(args.base_tokeniser, project_root)
  logger.info(f"[Kestrel] Loading base tokeniser: {base_path}")
  base_tok = AutoTokenizer.from_pretrained(base_path)

  results: Dict[str, Any] = {
    "base_tokeniser": {
      "name_or_path": base_path,
      "vocab_size": len(base_tok),
      "languages": {},
      "creole_parity_ratio": compute_parity_ratio(base_tok, aligned_pairs),
      "diagnostics": run_diagnostic_samples(base_tok),
    }
  }

  for lang, sents in lang_sentences.items():
    metrics = evaluate_language_corpus(base_tok, lang, sents)
    results["base_tokeniser"]["languages"][lang] = asdict(metrics)

  # ==================================================
  # Optional custom tokeniser evaluation & comparison
  # ==================================================

  if args.custom_tokeniser:
    custom_path = resolve_tokeniser_path(args.custom_tokeniser, project_root)
    logger.info(f"[Kestrel] Loading custom tokeniser: {custom_path}")
    custom_tok = AutoTokenizer.from_pretrained(custom_path)

    results["custom_tokeniser"] = {
      "name_or_path": custom_path,
      "vocab_size": len(custom_tok),
      "languages": {},
      "creole_parity_ratio": compute_parity_ratio(custom_tok, aligned_pairs),
      "diagnostics": run_diagnostic_samples(custom_tok),
    }
    results["comparison"] = {}

    for lang, sents in lang_sentences.items():
      c_metrics = evaluate_language_corpus(custom_tok, lang, sents)
      b_metrics_dict = results["base_tokeniser"]["languages"][lang]
      results["custom_tokeniser"]["languages"][lang] = asdict(c_metrics)

      # Token Saving Ratio (TSR) - Raj et al. (2025)
      base_tokens = b_metrics_dict["total_seq_tokens"]
      tsr = ((base_tokens - c_metrics.total_seq_tokens) / max(1, base_tokens)) * 100.0
      fertility_delta = c_metrics.subword_fertility - b_metrics_dict["subword_fertility"]
      cwr_delta = c_metrics.continued_word_rate - b_metrics_dict["continued_word_rate"]

      results["comparison"][lang] = {
        "token_saving_ratio_pct": round(tsr, 2),
        "subword_fertility_delta": round(fertility_delta, 4),
        "continued_word_rate_delta_pct": round(cwr_delta, 2),
      }

  # ==================================================
  # Determine output filename
  # ==================================================

  output_filename = args.output_file
  if not output_filename:
    # Use custom tokeniser path if evaluated, otherwise base tokeniser path
    target_path_for_naming = args.custom_tokeniser if args.custom_tokeniser else args.base_tokeniser
    output_filename = infer_output_filename(target_path_for_naming)

  # ==================================================
  # Save output to evaluation/ subfolder
  # ==================================================

  out_path = os.path.join(project_root, "evaluation", output_filename)
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
  logger.info(f"\n[Kestrel] Tokeniser benchmark saved to {out_path}")

if __name__ == "__main__":
  main()