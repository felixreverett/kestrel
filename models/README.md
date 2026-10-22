# kestrel/models/

Output subdirectory for merged models. These are ignored by default to avoid bloat, but can be found on Hugging Face [here](https://huggingface.co/felixreverett/kestrel)

## Kestrel version 1

These are the original fine-tuned models for translating French/English to and from Mauritian Creole.

There are four model variants: ones for both English and French, and ones for both 3B and 8B parameter sizes.

The models are:

| Model           | Notes
| :-------------- | :--
| Kestrel-1-3b-en | Available on HF as 16-bit nf and 4-bit quantised .gguf files.
| Kestrel-1-8b-en | Available on HF as 16-bit nf and 4-bit quantised .gguf files.
| Kestrel-1-3b-fr | Available on HF as 16-bit nf and 4-bit quantised .gguf files.
| Kestrel-1-8b-fr | Available on HF as 16-bit nf and 4-bit quantised .gguf files.

Formerly titled `kestrel-4.1` before successive iterations were developed.

## Kestrel version 2

These are the second iteration of Kestrel models, with a bespoke tokeniser trained on the KreolMorisienMT dataset to add 1,000 tokens. The tokeniser dropped the fertility rate of Mauritian Creole from `1.62` to `1.18`, demonstrating that the new base model was much more capable of tokenising Mauritian Creole writing.

These models were developed after the completion of my MSc thesis to explore a hypothesis I did not have time to pursue during the time limitations of the summer term: *To what extent could a custom tokeniser strengthen the models trained in a low-resource setting?*

Translation metric results (BLEU and chrF++) for these models can be seen on the main README at the root of this repository, but as a summary, the results were worse than expected, with some scores lower than that of the version 1 models. It is possible that the models either overfitted the training data, or experienced catastrophic forgetting, or a combination of both.

If/when I can be bothered, I'll look into this more. Train time took about a week.

The models (including the base with custom tokeniser) are:

| Model             | Notes
| :---------------- | :--
| Granite-4.1-3b-tk | The Granite base model with custom tokeniser (+1000 KM tokens).
| Granite-4.1-8b-tk | The Granite base model with custom tokeniser (+1000 KM tokens).
| Kestrel-2-3b-en   | Not yet available on HF.
| Kestrel-2-8b-en   | Not yet available on HF.
| Kestrel-2-3b-fr   | Not yet available on HF.
| Kestrel-2-8b-fr   | Not yet available on HF.