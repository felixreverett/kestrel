package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Messages struct {
	M []Message `json:"messages"`
}

func generateJSONL(assistantDataset []string, assistantLanguage string, userDataset []string, userLanguage string, outputPath string) bool {

	if len(assistantDataset) != len(userDataset) {
		fmt.Println("Error: dataset lengths mismatch.")
		return false
	}

	file, err := os.Create(outputPath)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return false
	}

	defer file.Close()

	systemPrompt := fmt.Sprintf("You are an expert translator. Translate the following %s text into %s", userLanguage, assistantLanguage)

	for i := 0; i < len(assistantDataset); i++ {
		conversation := Messages{
			M: []Message{
				{Role: "system", Content: systemPrompt},
				{Role: "assistant", Content: assistantDataset[i]},
				{Role: "user", Content: userDataset[i]},
			},
		}

		jsonData, err := json.Marshal(conversation)

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
	outputPath := flag.String("o", "output.jsonl", "Output path.")
	flag.Parse()

	// Ensure inputs are valid and of the same length
	validatedAssistantDataset, err := validateInput(*assistantDatasetPath)

	if err != nil {
		log.Fatalf("Failed to validate 'Assistant' dataset input: %s\n", err)
	}

	validatedUserDataset, err := validateInput(*userDatasetPath)

	if err != nil {
		log.Fatalf("Failed to validate 'User' dataset input: %s\n", err)
	}

	success := generateJSONL(validatedAssistantDataset, *assistantLanguage, validatedUserDataset, *userLanguage, *outputPath)

	if success {
		fmt.Println("Successfully generated:", *outputPath)
	}
}
