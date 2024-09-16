package main

import (
	"dense"
	"fmt"
	"path/filepath"
)

// Entry point of the program
func main() {
	// Step 1: Define project parameters for MNIST dataset and AI model testing
	projectName := "AIModelTestProject"
	inputSize := 28 * 28 // Example input size for MNIST (28x28 pixel images)
	outputSize := 10     // Example output size for MNIST digits (0-9)
	outputTypes := []string{"softmax"} // Use softmax for classification
	modelLocation := "models"
	methods := []string{"HillClimb"}          // Define the optimization method
	layerTypes := []string{"FFNN", "CNN", "LSTM"} // Define types of layers to test
	numModels := 100                     // Number of models to create and test
	cycleAllMutations := true          // Flag to cycle through all mutations
	topX := 3                          // Number of top models to track
	loadFilePath := ""                 // Load from file if needed

	// Step 2: Create the models folder if it doesn't exist
	err := dense.CreateDirectory(modelLocation)
	if err != nil {
		fmt.Printf("Error creating models folder: %v\n", err)
		return
	}

	// Step 3: Initialize the AIModelManager
	manager := &dense.AIModelManager{}
	manager.Init(projectName, inputSize, outputSize, outputTypes, modelLocation, methods, layerTypes, numModels, cycleAllMutations, topX, loadFilePath)

	// Step 4: Create the first generation of models and save them
	generationFolder := filepath.Join(modelLocation, "0")
	err = dense.CreateDirectory(generationFolder)
	if err != nil {
		fmt.Printf("Error creating generation folder: %v\n", err)
		return
	}

	// Step 5: Load the MNIST dataset (downloading if necessary)
	err = dense.EnsureMNISTDownloads()
	if err != nil {
		fmt.Printf("Error downloading MNIST data: %v\n", err)
		return
	}
	mnist, err := dense.LoadMNISTOLD() // Loading the MNIST dataset
	if err != nil {
		fmt.Printf("Error loading MNIST: %v\n", err)
		return
	}

	// Split the MNIST data into training and testing sets
	_, testData := splitData(mnist)

	// Step 6: Create the first generation of models
	manager.CreateFirstGeneration(generationFolder)
	fmt.Println("First generation of models created.")

	// Step 7: Evaluate each model after applying mutations and report accuracy
	for i := 0; i < numModels; i++ {
		modelFile := filepath.Join(generationFolder, fmt.Sprintf("model-%d.json", i+1))
		config, err := dense.LoadNetworkFromFile(modelFile)
		if err != nil {
			fmt.Printf("Error loading model %d: %v\n", i+1, err)
			continue
		}

		// Apply mutations to the models before testing
		learningRate := 0.01
		mutationRate := 20
		manager.ApplyAllMutations(config, learningRate, mutationRate)

		// Evaluate the fitness of the model on the test dataset
		fitness := evaluateFitness(config, testData)
		fmt.Printf("Model %d accuracy: %.4f%%\n", i+1, fitness*100)

		// Save the mutated model
		mutatedModelFile := filepath.Join(generationFolder, fmt.Sprintf("mutated-model-%d.json", i+1))
		err = dense.SaveNetworkConfig(config, mutatedModelFile)
		if err != nil {
			fmt.Printf("Error saving mutated model %d: %v\n", i+1, err)
		} else {
			fmt.Printf("Mutated model %d saved to %s\n", i+1, mutatedModelFile)
		}
	}

	fmt.Println("All models evaluated and mutated versions saved.")
}

// Function to split the MNIST data into training (80%) and testing (20%)
func splitData(mnist *dense.MNISTData) (trainData, testData *dense.MNISTData) {
	totalImages := len(mnist.Images)
	splitIndex := int(float64(totalImages) * 0.8)

	trainData = &dense.MNISTData{
		Images: mnist.Images[:splitIndex],
		Labels: mnist.Labels[:splitIndex],
	}

	testData = &dense.MNISTData{
		Images: mnist.Images[splitIndex:],
		Labels: mnist.Labels[splitIndex:], // Fix here, use slice for remaining test data
	}

	return trainData, testData
}

// Evaluate the model's performance on the MNIST test dataset
func evaluateFitness(config *dense.NetworkConfig, mnist *dense.MNISTData) float64 {
	correct := 0
	total := len(mnist.Images)

	for i, image := range mnist.Images {
		// Prepare input data
		input := make(map[string]interface{})
		for j, pixel := range image {
			inputKey := fmt.Sprintf("input%d", j)
			input[inputKey] = float64(pixel) / 255.0 // Normalize pixel values
		}

		// Run the model's feedforward function
		outputs := dense.Feedforward(config, input)

		// Interpret the model output (e.g., predicted digit)
		predictedDigit := 0
		highestProb := 0.0
		for k := 0; k < 10; k++ {
			outputKey := fmt.Sprintf("output%d", k)
			if prob, ok := outputs[outputKey]; ok && prob > highestProb {
				highestProb = prob
				predictedDigit = k
			}
		}

		expectedDigit := int(mnist.Labels[i]) // Use the byte value directly
		if predictedDigit == expectedDigit {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total)
	return accuracy
}
