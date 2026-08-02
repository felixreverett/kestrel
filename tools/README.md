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