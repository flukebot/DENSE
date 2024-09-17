package main

import (
	"dense"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
)

// ModelEvaluation represents the structure holding model and its accuracy for easy sorting.
type ModelEvaluation struct {
	Model    *dense.NetworkConfig
	Accuracy float64
}

// LoadMNISTData loads the actual MNIST dataset.
func LoadMNISTData() (*dense.MNISTData, error) {
	// Ensure the dataset is downloaded
	err := dense.EnsureMNISTDownloads()
	if err != nil {
		return nil, fmt.Errorf("failed to ensure MNIST downloads: %v", err)
	}

	// Load the dataset
	mnistData, err := dense.LoadMNISTOLD()
	if err != nil {
		return nil, fmt.Errorf("failed to load MNIST dataset: %v", err)
	}

	return mnistData, nil
}

// EvaluateModel evaluates the model's fitness using MNIST data and returns accuracy.
func EvaluateModel(model *dense.NetworkConfig, mnistData *dense.MNISTData) float64 {
	totalCorrect := 0
	totalImages := len(mnistData.Images)

	// Loop through the MNIST data to make predictions
	for i := 0; i < totalImages; i++ {
		// Get the input image and its corresponding label
		inputImage := mnistData.Images[i] // The image data (flattened array of bytes)
		trueLabel := mnistData.Labels[i]  // The correct label (0-9)

		// Prepare the input data for the neural network (convert byte to float64)
		inputData := make(map[string]interface{})
		for j, pixel := range inputImage {
			inputData[fmt.Sprintf("input%d", j)] = float64(pixel) / 255.0 // Normalize pixel values
		}

		// Perform a forward pass through the network
		predictedOutput := dense.Feedforward(model, inputData)

		// Determine the predicted label (e.g., argmax on the output layer)
		predictedLabel := ArgMax(predictedOutput)

		// Check if the predicted label matches the true label
		if predictedLabel == int(trueLabel) {
			totalCorrect++
		}
	}

	// Calculate accuracy: number of correct predictions / total number of images
	accuracy := float64(totalCorrect) / float64(totalImages)

	return accuracy
}

// ArgMax returns the index of the largest value in the output layer (predicted label).
func ArgMax(output map[string]float64) int {
	maxValue := -1.0
	maxIndex := -1
	for key, value := range output {
		// Extract the index from the key (e.g., "output0", "output1", ...)
		var index int
		fmt.Sscanf(key, "output%d", &index)
		if value > maxValue {
			maxValue = value
			maxIndex = index
		}
	}
	return maxIndex
}

// Main function: orchestrates the evolutionary process for generating models.
func main() {
	// Define project parameters
	projectName := "AIModelTestProject"
	inputSize := 28 * 28 // Input size for MNIST data
	outputSize := 10      // Output size for MNIST digits (0-9)
	outputTypes := []string{"softmax"} // Activation type for output layer
	modelLocation := "models"           // Folder to store models and generations
	numModels := 10                     // Number of models per generation
	topX := 10                          // Top 10 models to copy to the next generation
	numGenerations := 100               // Total number of generations
	mutationRate := 10                  // Mutation rate for models

	// Load actual MNIST data
	mnistData, err := LoadMNISTData()
	if err != nil {
		fmt.Printf("Error loading MNIST data: %v\n", err)
		return
	}

	// Create generation 0 with all random models
	generation0 := make([]*dense.NetworkConfig, numModels)
	for i := 0; i < numModels; i++ {
		modelConfig := dense.CreateRandomNetworkConfig(inputSize, outputSize, outputTypes, fmt.Sprintf("model%d", i), projectName)
		dense.MutateNetwork(modelConfig, 0.01, mutationRate) // Apply mutations
		generation0[i] = modelConfig
	}

	// Evaluate models in generation 0
	modelEvaluations := evaluateModels(generation0, mnistData)

	// Loop over generations, starting from generation 1
	for gen := 1; gen < numGenerations; gen++ {
		fmt.Printf("Starting Generation %d...\n", gen)
		generationFolder := filepath.Join(modelLocation, fmt.Sprintf("%d", gen))
		if err := os.MkdirAll(generationFolder, os.ModePerm); err != nil {
			fmt.Printf("Error creating folder for generation %d: %v\n", gen, err)
			return
		}

		// Copy the top 10 best-performing models without mutations into the next generation
		nextGeneration := make([]*dense.NetworkConfig, 0, numModels)
		for _, modelEval := range modelEvaluations[:topX] {
			nextGeneration = append(nextGeneration, modelEval.Model)
		}

		// Fill the rest of the next generation by mutating the top 10 models
		for len(nextGeneration) < numModels {
			// Select a random model from the top 10
			originalModel := modelEvaluations[rand.Intn(topX)].Model
			// Create a mutated copy
			mutatedModel := dense.DeepCopy(originalModel)
			dense.MutateNetwork(mutatedModel, 0.01, mutationRate)
			nextGeneration = append(nextGeneration, mutatedModel)
		}

		// Evaluate the new generation
		modelEvaluations = evaluateModels(nextGeneration, mnistData)

		// Save the models in the new generation
		for i, model := range nextGeneration {
			modelPath := filepath.Join(generationFolder, fmt.Sprintf("model%d.json", i))
			if err := dense.SaveNetworkConfig(modelPath, model); err != nil {
				fmt.Printf("Error saving model %d in generation %d: %v\n", i, gen, err)
				return
			}
		}

		// Print the top model's accuracy for this generation
		if len(modelEvaluations) > 0 {
			fmt.Printf("Top model accuracy for generation %d: %.4f\n", gen, modelEvaluations[0].Accuracy)
		}
	}

	fmt.Println("Evolution complete.")
}

// evaluateModels evaluates a slice of models and returns their evaluations sorted by accuracy
func evaluateModels(models []*dense.NetworkConfig, mnistData *dense.MNISTData) []ModelEvaluation {
	modelEvaluations := make([]ModelEvaluation, len(models))
	for i, model := range models {
		accuracy := EvaluateModel(model, mnistData)
		modelEvaluations[i] = ModelEvaluation{
			Model:    model,
			Accuracy: accuracy,
		}
	}

	// Sort models by accuracy in descending order
	sort.Slice(modelEvaluations, func(i, j int) bool {
		return modelEvaluations[i].Accuracy > modelEvaluations[j].Accuracy
	})

	return modelEvaluations
}