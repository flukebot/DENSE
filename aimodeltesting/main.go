package main

import (
	"dense"
	"fmt"
	"path/filepath"
)

func main() {
	// Define project parameters
	projectName := "AIModelTestProject"
	inputSize := 28 * 28    // Example input size for MNIST data
	outputSize := 10        // Example output size for MNIST digits (0-9)
	outputTypes := []string{"softmax", "relu"} // Example activation types
	modelLocation := "models" // Folder to store models and generations
	methods := []string{"HillClimb", "NEAT"}   // Define training methods (HillClimb and NEAT in this case)
	layerTypes := []string{"FFNN", "CNN"}      // Define layer types (FFNN, CNN, etc.)
	numModels := 5                             // Number of models to start with
	cycleAllMutations := true                  // Cycle through all mutations at the start
	topX := 3                                  // Number of top models to track per generation
	loadFilePath := ""                         // No load file for this test, starting fresh

	// Create the "models" folder if it doesn't exist
	err := dense.CreateDirectory(modelLocation)
	if err != nil {
		fmt.Printf("Error creating models folder: %v\n", err)
		return
	}

	// Create AIModelManager and initialize it
	manager := &dense.AIModelManager{}
	manager.Init(projectName, inputSize, outputSize, outputTypes, modelLocation, methods, layerTypes, numModels, cycleAllMutations, topX, loadFilePath)

	// Create a generation_0 folder inside models folder for generation 0
	generationFolder := filepath.Join(modelLocation, "generation_0")
	err = dense.CreateDirectory(generationFolder)
	if err != nil {
		fmt.Printf("Error creating generation_0 folder: %v\n", err)
		return
	}
	fmt.Printf("Created folder for generation 0: %s\n", generationFolder)

	// Placeholder for other initialization steps if needed later
	fmt.Println("Project setup complete.")
}
