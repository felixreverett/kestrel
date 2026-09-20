package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Messages struct {
	M []Message `json:"messages"`
}

func generateJSONL(
	assistantDataset []string, assistantLanguage string,
	userDataset []string, userLanguage string,
	outputPath string, bidirectional bool, shuffle bool, promptLanguage string) bool {

	if len(assistantDataset) != len(userDataset) {
		fmt.Println("Error: dataset lengths mismatch.")
		return false
	}

	var promptForward, promptReverse string

	if promptLanguage == "fr" {
		promptForward = fmt.Sprintf("Vous êtes un traducteur expert. Traduisez le texte %s suivant en %s.", userLanguage, assistantLanguage)
		promptReverse = fmt.Sprintf("Vous êtes un traducteur expert. Traduisez le texte %s suivant en %s.", assistantLanguage, userLanguage)
	} else {
		promptForward = fmt.Sprintf("You are an expert translator. Translate the following %s text into %s", userLanguage, assistantLanguage)
		promptReverse = fmt.Sprintf("You are an expert translator. Translate the following %s text into %s", assistantLanguage, userLanguage)
	}

	var conversations []Messages

	for i := 0; i < len(assistantDataset); i++ {
		conversations = append(conversations, Messages{
			M: []Message{
				{Role: "system", Content: promptForward},
				{Role: "user", Content: userDataset[i]},
				{Role: "assistant", Content: assistantDataset[i]},
			},
		})

		if bidirectional {
			conversations = append(conversations, Messages{
				M: []Message{
					{Role: "system", Content: promptReverse},
					{Role: "user", Content: assistantDataset[i]},
					{Role: "assistant", Content: userDataset[i]},
				},
			})
		}
	}

	if shuffle {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		r.Shuffle(len(conversations), func(i, j int) {
			conversations[i], conversations[j] = conversations[j], conversations[i]
		})
	}

	file, err := os.Create(outputPath)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return false
	}
	defer file.Close()

	for _, conv := range conversations {
		jsonData, err := json.Marshal(conv)
		if err != nil {
			fmt.Println("Error marshalling JSON: ", err)
			return false
		}

		_, err = file.WriteString(string(jsonData) + "\n")
		if err != nil {
			fmt.Println("Error writing to file: ", err)
			return false
		}
	}

	return true
}

func validateInput(filePath string) ([]string, error) {

	content, err := os.ReadFile("datasets/" + filePath)

	if err != nil {
		return nil, err
	}

	trimmedInput := strings.ReplaceAll(string(content), "\r", "")
	trimmedInput = strings.TrimSuffix(trimmedInput, "\n")
	fileLines := strings.Split(trimmedInput, "\n")

	return fileLines, nil
}

func main() {

	// Flags
	assistantDatasetPath := flag.String("ad", "MorisienMT/dev.en", "The Assistant Dataset.")
	assistantLanguage := flag.String("al", "English", "The Assistant Language.")
	userDatasetPath := flag.String("ud", "MorisienMT/dev.cr", "The User Dataset.")
	userLanguage := flag.String("ul", "Mauritian Creole", "The User Language.")
	bidirectional := flag.Bool("b", true, "Generate translations in both directions")
	shuffle := flag.Bool("s", true, "Shuffle the resulting dataset")
	promptLanguage := flag.String("pl", "en", "The language of the system prompt (en or fr).")
	outputPath := flag.String("o", "", "Output path.")

	flag.Parse()

	var finalOutputPath string
	if *outputPath == "" {
		var outputType string
		if *bidirectional {
			outputType = "bidirectional"
		} else {
			outputType = "unidirectional"
		}
		finalOutputPath = fmt.Sprintf("datasets/%s-%s-%s.jsonl", *assistantLanguage, *userLanguage, outputType)
	} else {
		finalOutputPath = *outputPath
	}

	// Ensure inputs are valid and of the same length
	validatedAssistantDataset, err := validateInput(*assistantDatasetPath)

	if err != nil {
		log.Fatalf("Failed to validate 'Assistant' dataset input: %s\n", err)
	}

	validatedUserDataset, err := validateInput(*userDatasetPath)

	if err != nil {
		log.Fatalf("Failed to validate 'User' dataset input: %s\n", err)
	}

	success := generateJSONL(
		validatedAssistantDataset, *assistantLanguage,
		validatedUserDataset, *userLanguage,
		finalOutputPath, *bidirectional, *shuffle, *promptLanguage,
	)

	if success {
		fmt.Println("Successfully generated:", finalOutputPath)
	}
}
