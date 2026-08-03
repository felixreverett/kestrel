# kestrel/tools/

Subfolder for assistive tools as part of the local language model development process.

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

## 2 train.py

## 3 merge.py

### 3.1 What is merge.py?

The merge script is used to merge trained adapter weights back into the base model as part of the fine-tuning pipeline. As such, the train script must first be used to produce an adapter.

### 3.2 Usage instructions

To merge your adapter weights back into the base model, first ensure your python environment is activated. If not, activate it with:

`conda activate kestrel`

Next, run the script with the below command. The script will look for a `kestrel_config.json` file within the adapter's directory to automatically detect the model, meaning -m is not required.

`python tools/merge.py {-m 4.1-3b} -a kestrel-4.1-3b -o kestrel-4.1-3b-final`

### 3.3 Command-line flags

| Command line Flag | Default Value                 | Comments
| :---------------- | :---------------------------- | :--
| -m                | "4.1-8b"                      | The base model. Should match that used for training
| -a                | "kestrel-4.1-8b"              | The adapter weights to merge back into the base model
| -o                | "kestrel-4.1-8b-final"        | The final output folder name for the models/ subdirectory