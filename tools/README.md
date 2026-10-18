# kestrel/tools/

Subfolder for assistive tools as part of the local language model development process.

## Contents
1. **inputFormatter.go**: Converts datasets into ChatML format.
2. **train.py**: Fine-tunes a new model.
3. **merge.py**: Merges fine-tuned adapter weights.
4. **eval.py**: Calculates BLEU and chrF++ scores of a given model.

<!--- ============= -->

## 1 inputFormatter

### 1.1 Build instructions

To build the executable, ensure you have **Go** installed, and input `go build -o tools/formatter.exe tools/inputFormatter.go` into the command line (if you are on Windows).

Execute the file with `tools/formatter.exe` plus any command line flags as specified in 1.3 below.

### 1.2 Executing as a script

If you prefer to use Go as a script, ensure you have **Go** installed and input `go run tools/inputFormatter.go` into the command line plus any command line flags as specified in 1.3 below.

### 1.3 Command-line flags

There are several command line flags which can be seen by adding the flag `-help`, or can be viewed in the table below:

| Command line Flag | Default Value                 | Comments
| :---------------- | :---------------------------- | :--
| -ad               | "MorisienMT/dev.en"           | Assistant dataset
| -al               | "English"                     | Assistant language. Used to set the system prompt and output name.
| -ud               | "MorisienMT/dev.cr            | User dataset
| -ul               | "Mauritian Creole"            | User language. Used to set the system prompt and output name.
| -b                | true                          | Bidirectional. Whether to generate prompts in both directions, or just user->assistant.
| -o                | "datasets/{al}{ul}{b}.jsonl"  | Output name and path. Defaults to using the Assistant and User languages + whether the resulting data is bidirectional.
| -s                | true                          | Whether to shuffle the output. Defaults to true.

### 1.4 Example flags

To recreate the .jsonl datasets used in training, enter the following flags with your command:
`-ad MorisienMT/train.en-cr.en -al "English" -ud MorisienMT/train.en-cr.cr -ul "Mauritian Creole" -b=true`

`-ad MorisienMT/train.fr-cr.fr -al "French" -ud MorisienMT/train.fr-cr.cr -ul "Mauritian Creole" -b=true`

<!--- ============= -->

## 2 train.py

### 2.1 What is train.py?

The train script is used to fine-tune an existing base model on a provided dataset.

### 2.2 Instructions

To fine-tune the base model on the dataset, first ensure your python environment is activated.

Next, run the script with `python tools/train.py` plus any additional command-line flags. It is recommended
to simply run the script with a predefined JSON config for ease of use: `python tools/train.py -c exampleconfig.json`. If a config is provided, all other command-line flags will be ignored.

#### 2.2.1 Example command-line statements
1. `python tools/train.py -c configs/kestrel-1-3b-en-config.json`
2. `python tools/train.py -c configs/kestrel-1-3b-fr-config.json`
3. `python tools/train.py -c configs/kestrel-1-8b-en-config.json`
4. `python tools/train.py -c configs/kestrel-1-8b-fr-config.json`
5. `python tools/train.py -c configs/kestrel-2-3b-en-config.json`
6. `python tools/train.py -c configs/kestrel-2-3b-fr-config.json`
7. `python tools/train.py -c configs/kestrel-2-8b-en-config.json`
8. `python tools/train.py -c configs/kestrel-2-8b-fr-config.json`

### 2.3 Command-line flags

| Command line Flag | Default Value     | Comments
| :---------------- | :---------------- | :---------
| -c / --config     | None              | Config file, (located in configs/)
| -m / --model      | "4.1-3b"          | Base model shortcut.
| -d / --dataset    | English-Mauritian Creole-bidirectional.jsonl | Dataset filename (located in datasets/)
| -o / --output     | None              | Output filename (located in adapters/)
| -bs / --batchsize | 2                 | Per-device training batch size
| -ms / --maxsteps  | 1                 | Maximum training steps (or -1 for full epochs)

<!--- ============= -->

## 3 merge.py

### 3.1 What is merge.py?

The merge script is used to merge trained adapter weights back into the base model as part of the fine-tuning pipeline. As such, the train script must first be used to produce an adapter.

### 3.2 Usage instructions

To merge your adapter weights back into the base model, first ensure your python environment is activated. If not, activate it with:

`conda activate kestrel`

Next, run the script with the below command. The script will look for a `kestrel_config.json` file within the adapter's directory to automatically detect the model, meaning -m is not required.

`python tools/merge.py -a kestrel-4.1-3b`

#### 3.2.1 Example command-line statements
1. `python tools/merge.py -a kestrel-4.1-3b-en`
2. `python tools/merge.py -a kestrel-4.1-3b-fr`
3. `python tools/merge.py -a kestrel-4.1-8b-en`
4. `python tools/merge.py -a kestrel-4.1-8b-fr`
5. `python tools/merge.py -a kestrel-2-3b-en`
6. `python tools/merge.py -a kestrel-2-3b-fr`
7. `python tools/merge.py -a kestrel-2-8b-en`
8. `python tools/merge.py -a kestrel-2-8b-fr`

### 3.3 Command-line flags

| Command line Flag | Default Value | Comments
| :---------------- | :------------ | :--------------------------------------------------
| -a / --adapter    | None          | The directory name of the trained adapter
| -m / --model      | None          | The base model. Should match that used for training
| -o / --output     | None          | Output folder name for the models/ subdirectory

## 4 eval.py

The evaluation script `eval.py` is used to test the BLEU and chrF++ metrics for a fine-tuned model, specifically one in Hugging-Face format (rather than a .gguf). BLEU and chrF++ are two similar machine translation scores, with the former being considerably more established, but the latter is arguably more in-line with human evaluations.

I highly recommend reading [Papineni et al., 2002](https://aclanthology.org/P02-1040/) and [Popovic, 2017](https://aclanthology.org/W17-4770/) if you wish to learn more. See [Mathur et al., 2020](https://aclanthology.org/2020.acl-main.448/) for a review of BLEU's shortcomings.

As with all helper scripts in Kestrel, this is designed to be highly-extensible. There is an in-memory config file for greater customisability.

### 4.1 Instructions

Explicit commands are provided in this section. Note that the script is designed to be extensible, and custom configurations are supported. These commands are provided to facilitate a 1:1 development.

#### 4.1.1 Evaluate Base Models
1. `python tools/eval.py -m granite-4.1-3b -t "EN-KM-test.jsonl" -o results_granite-4.1-3b-en.json`
2. `python tools/eval.py -m granite-4.1-3b -t "FR-KM-test.jsonl" -o results_granite-4.1-3b-fr.json`
3. `python tools/eval.py -m granite-4.1-8b -t "EN-KM-test.jsonl" -o results_granite-4.1-8b-en.json --load_in_8bit`
4. `python tools/eval.py -m granite-4.1-8b -t "FR-KM-test.jsonl" -o results_granite-4.1-8b-fr.json --load_in_8bit`

#### 4.1.2 Evaluate Tokenised Base Models (not recommended)
These models are still zero-shot and results are not valuable for analysis. Experimental testing produced largely meaningless outputs.
1. `python tools/eval.py -m granite-4.1-3b-tk -t "EN-KM-test.jsonl" -o results_granite-4.1-3b-tk-en.json`
2. `python tools/eval.py -m granite-4.1-3b-tk -t "FR-KM-test.jsonl" -o results_granite-4.1-3b-tk-fr.json`
3. `python tools/eval.py -m granite-4.1-8b-tk -t "EN-KM-test.jsonl" -o results_granite-4.1-8b-tk-en.json --load_in_8bit`
4. `python tools/eval.py -m granite-4.1-8b-tk -t "FR-KM-test.jsonl" -o results_granite-4.1-8b-tk-fr.json --load_in_8bit`

#### 4.1.3 Evaluate Fine-tuned Kestrel 1 models
1. `python tools/eval.py -m kestrel-1-3b-en -t "EN-KM-test.jsonl" -o results_kestrel-1-3b-en.json`
2. `python tools/eval.py -m kestrel-1-3b-fr -t "FR-KM-test.jsonl" -o results_kestrel-1-3b-fn.json`
3. `python tools/eval.py -m kestrel-1-8b-en -t "EN-KM-test.jsonl" -o results_kestrel-1-8b-en.json --load_in_8bit`
4. `python tools/eval.py -m kestrel-1-8b-fr -t "FR-KM-test.jsonl" -o results_kestrel-1-8b-fn.json --load_in_8bit`

#### 4.1.4 Evaluate Fine-tuned Kestrel 2 models
1. `python tools/eval.py -m kestrel-2-3b-en -t "EN-KM-test.jsonl" -o results_kestrel-2-3b-en.json`
2. `python tools/eval.py -m kestrel-2-3b-fr -t "FR-KM-test.jsonl" -o results_kestrel-2-3b-fn.json`
3. `python tools/eval.py -m kestrel-2-8b-en -t "EN-KM-test.jsonl" -o results_kestrel-2-8b-en.json --load_in_8bit`
4. `python tools/eval.py -m kestrel-2-8b-fr -t "FR-KM-test.jsonl" -o results_kestrel-2-8b-fn.json --load_in_8bit`

### 4.2 Usage instructions

To test a completed model, activate your Python environment `conda activate kestrel`, then execute the script with `python tools/eval.py -m {model_name} -t {test_filename} -o {output_name}`. All command-line flags are folder-relative. For example, to test a model named `kestrel-4.1-3b-final`, one could type the command `python tools.eval.py -m kestrel-4.1-3b-final -t "English-Mauritian Creole-test.jsonl" -o eval_results_kestrel-4.1-3b-final.json`

### 4.3 Command-line flags

| Command line Flag | Default Value     | Comments
| :---------------- | :---------------- | :--------------------------------------------------
| -m / --model_path | N/A (Required)    | The model to evaluate. Found in `models/`
| -t / --test_file  | eval_results.json | Test dataset. Found in `datasets/`
| -o / --output_path| N/A (Required)    | JSON output filename. Saved in `evaluation/`
| --max_samples     | None              | Optional limit on test samples
| --load_in_8bit    | False             | Load model in 8bit
| --load_in_4bit    | False             | Load model in 4bit

## 5 Converting to GGUF with llama.cpp

### 5.1 Instructions
1. Clone llama.cpp from GitHub
2. Set working directory to .../llama.cpp
3. (Recommended) Use Python env
    a. activate. e.g.: `conda activate kestrel`
    b. Install requirements ` pip install -r requirements/requirements-convert_hf_to_gguf.txt`
4. Convert merged fine-tuned models to .gguf format
    a. `python convert_hf_to_gguf.py ../kestrel/models/kestrel-4.1-3b-final --outfile ../kestrel/models/kestrel-4.1-8b-f16.gguf --outtype f16`
5. (Recommended) quantise to 4-bit using llama-quantize (requires compiling binary with cmake)

## 6 tk_fertility_eval.py

This script is used to calculate the fertility rate of the base and custom tokenisers, casting insight on the intrinsic tokenisation metrics. A custom tokeniser is hypothesised to produce a better final model, though this needs to be evaluated separately using extrinsic metrics (BLEU and chrF++).

### 6.1 Instructions

#### 6.1.1 Evaluate base tokeniser (3B and 8B in both FR and EN)
3B and 8B models use the same tokeniser, so results are identical between 3B/8B versions.
1. `python tools/tk_fertility-eval.py -b 4.1-3b -d "English-Mauritian Creole-test.jsonl" -o "tk_base-4.1-3b-en.json"`
2. `python tools/tk_fertility-eval.py -b 4.1-3b -d "French-Mauritian Creole-test.jsonl" -o "tk_base-4.1-3b-fr.json"`
3. `python tools/tk_fertility-eval.py -b 4.1-8b -d "English-Mauritian Creole-test.jsonl" -o "tk_base-4.1-8b-en.json"`
4. `python tools/tk_fertility-eval.py -b 4.1-8b -d "French-Mauritian Creole-test.jsonl" -o "tk_base-4.1-8b-fr.json"`

#### 6.1.2 Evaluate custom tokeniser
1. `python tools/tk_fertility-eval.py -b 4.1-3b -c models/granite-4.1-3b-tk -d "English-Mauritian Creole-test.jsonl" -o "tk_tk-4.1-3b-en.json"`

## 7 tk_build-tokeniser.py

This script is used to produce a custom tokeniser for Mauritian Creole to be used for Kestrel from versions 2.0 onwards.

### 7.1 Instructions
1. `python tools/tk_build-tokeniser.py -b 4.1-3b -o granite-4.1-3b-tk`
2. `python tools/tk_build-tokeniser.py -b 4.1-8b -o granite-4.1-8b-tk`