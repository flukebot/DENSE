package main

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"dense" // Ensure this import path matches your project structure
)

type Result struct {
	fitness  float64
	tmpModel string
}

// Function to split the MNIST data into training (80%) and testing (20%)
func splitData(mnist *dense.MNISTData) (trainData, testData *dense.MNISTData) {
	totalImages := len(mnist.Images)
	splitIndex := int(float64(totalImages) * 0.8)

	trainData = &dense.MNISTData{
		Images: mnist.Images[:splitIndex],
		Labels: mnist.Labels[:splitIndex], // Fix: properly split Labels as a slice
	}

	testData = &dense.MNISTData{
		Images: mnist.Images[splitIndex:],
		Labels: mnist.Labels[splitIndex:], // Fix: properly split Labels as a slice
	}

	return trainData, testData
}

// Evaluates the fitness of the model using the provided dataset
func evaluateFitness(config *dense.NetworkConfig, mnist *dense.MNISTData) float64 {
	correct := 0
	total := len(mnist.Images)

	for i, image := range mnist.Images {
		// Prepare input data
		input := make(map[string]interface{})
		for j, pixel := range image {
			inputKey := fmt.Sprintf("input%d", j)
			input[inputKey] = float64(pixel) / 255.0
		}

		// Run feedforward
		outputs := dense.Feedforward(config, input)

		// Interpret the output
		predictedDigit := 0
		highestProb := 0.0
		for k := 0; k < 10; k++ {
			outputKey := fmt.Sprintf("output%d", k)
			if prob, ok := outputs[outputKey]; ok && prob > highestProb {
				highestProb = prob
				predictedDigit = k
			}
		}

		expectedDigit := int(mnist.Labels[i])
		if predictedDigit == expectedDigit {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total)
	return accuracy
}

// Hill Climbing Optimization that combines FFNN, LSTM, and CNN
func hillClimbingOptimize(trainData, testData *dense.MNISTData, iterations, batchSize, numWorkers int, learningRate, fitnessBuffer float64) float64 {
	// Load the best config from the file
	bestConfig, err := dense.LoadNetworkFromFile("best_model.json")
	if err != nil {
		fmt.Println("Error loading best model, using default config:", err)
		return 0.0
	}

	// Initial accuracies
	bestTrainingAccuracy := bestConfig.Metadata.LastTrainingAccuracy
	bestTestAccuracy := bestConfig.Metadata.LastTestAccuracy
	fmt.Printf("Starting optimization with initial accuracies (Training: %.4f%%, Test: %.4f%%)\n", bestTrainingAccuracy*100, bestTestAccuracy*100)

	for i := 0; i < iterations; i += batchSize {
		var wg sync.WaitGroup
		results := make(chan Result, batchSize)

		// Start a batch of workers
		for j := 0; j < batchSize; j++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				tmpModelFilename := fmt.Sprintf("tmp_model_%d.json", workerID)

				// Load a temporary copy of the best model for this worker
				currentConfig := dense.DeepCopy(bestConfig)

				// Mutate the model and evaluate its fitness
				dense.MutateNetwork(currentConfig, learningRate, 50)
				newFitness := evaluateFitness(currentConfig, trainData)

				// Save mutated model to the temporary file
				dense.SaveNetworkToFile(currentConfig, tmpModelFilename)

				results <- Result{fitness: newFitness, tmpModel: tmpModelFilename}
			}(j % numWorkers) // Distribute jobs across workers
		}

		// Wait for the batch to complete
		go func() {
			wg.Wait()
			close(results)
		}()

		// Evaluate results
		for result := range results {
			if result.fitness > bestTrainingAccuracy+fitnessBuffer {
				bestTrainingAccuracy = result.fitness
				// Load the best temp model
				tmpConfig, err := dense.LoadNetworkFromFile(result.tmpModel)
				if err != nil {
					fmt.Println("Error loading temp model:", err)
					continue
				}
				bestConfig = tmpConfig
				fmt.Printf("New best model found with accuracy: %.4f%%\n", bestTrainingAccuracy*100)
			}

			// Delete the temporary model file
			err = os.Remove(result.tmpModel)
			if err != nil {
				fmt.Printf("Warning: Failed to delete temp model file: %s\n", result.tmpModel)
			}
		}

		fmt.Printf("\nBatch ending at iteration %d: Current best training accuracy: %.4f%%\n", i+batchSize, bestTrainingAccuracy*100)
	}

	// Update metadata with new evaluations
	bestConfig.Metadata.LastTrainingAccuracy = bestTrainingAccuracy
	bestConfig.Metadata.LastTestAccuracy = evaluateFitness(bestConfig, testData)
	fmt.Printf("Final model accuracy on test set: %.4f%%\n", bestConfig.Metadata.LastTestAccuracy*100)

	// Save the best model after all batches are complete
	dense.SaveNetworkToFile(bestConfig, "best_model.json")

	return bestTrainingAccuracy * 100
}

// Check if a file exists
func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

// Save the training and test data to files
func saveData(trainData, testData *dense.MNISTData, trainFile, testFile string) error {
	err := dense.SaveMNIST(trainFile, trainData)
	if err != nil {
		return err
	}
	err = dense.SaveMNIST(testFile, testData)
	return err
}

// Load the training and test data from files
func loadData(trainFile, testFile string) (trainData, testData *dense.MNISTData, err error) {
	trainData, err = dense.LoadMNIST(trainFile)
	if err != nil {
		return nil, nil, err
	}
	testData, err = dense.LoadMNIST(testFile)
	if err != nil {
		return nil, nil, err
	}
	return trainData, testData, nil
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Ensure MNIST dataset is downloaded
	err := dense.EnsureMNISTDownloads()
	if err != nil {
		fmt.Println("Error downloading MNIST dataset:", err)
		return
	}

	var trainData, testData *dense.MNISTData
	trainFile := "train_data.json"
	testFile := "test_data.json"

	// Check if the training and test data already exist
	if fileExists(trainFile) && fileExists(testFile) {
		// Load the saved training and test data
		fmt.Println("Loading saved training and test data...")
		trainData, testData, err = loadData(trainFile, testFile)
		if err != nil {
			fmt.Println("Error loading saved data:", err)
			return
		}
	} else {
		// Load the full MNIST data from raw files and split it into training and test sets
		mnist, err := dense.LoadMNISTOLD()
		if err != nil {
			fmt.Println("Error loading MNIST dataset:", err)
			return
		}
		trainData, testData = splitData(mnist)

		// Save the split data for future use
		err = saveData(trainData, testData, trainFile, testFile)
		if err != nil {
			fmt.Println("Error saving split data:", err)
			return
		}
	}

	// Check if the best model already exists
	if !fileExists("best_model.json") {
		fmt.Println("No existing model found, creating a new one...")
		numInputs := 28 * 28
		numOutputs := 10 // For digits 0-9
		outputActivationTypes := make([]string, numOutputs)
		for i := range outputActivationTypes {
			outputActivationTypes[i] = "softmax"
		}

		// Create a combined FFNN, LSTM, CNN model
		initialConfig := createCombinedModel(numInputs, numOutputs, outputActivationTypes)
		dense.SaveNetworkToFile(initialConfig, "best_model.json") // Save the initial model
	} else {
		fmt.Println("Loading existing best model from file...")
	}

	fmt.Println("Starting multithreaded hill climbing optimization...")

	desiredAccuracy := 80.0
	maxDuration := 12 * time.Hour
	startTime := time.Now()

	numWorkers := 4  // Adjust this based on your system's capabilities
	batchSize := 10  // Set the batch size

	for {
		bestFitness := hillClimbingOptimize(trainData, testData, 100, batchSize, numWorkers, 0.5, 0.001)
		fmt.Printf("Current best model accuracy (training set): %.4f%%\n", bestFitness)

		if bestFitness >= desiredAccuracy || time.Since(startTime) >= maxDuration {
			break
		}

		time.Sleep(1 * time.Second)
	}

	// Load and print the best saved model
	loadedConfig, err := dense.LoadNetworkFromFile("best_model.json")
	if err != nil {
		fmt.Println("Error loading final best model:", err)
		return
	}
	fmt.Println("Loaded best model configuration:", loadedConfig)
}

// Create a model combining FFNN, LSTM, and CNN layers
func createCombinedModel(numInputs, numOutputs int, outputActivationTypes []string) *dense.NetworkConfig {
	modelID := "combined_model"
	projectName := "FFNN + LSTM + CNN MNIST"

	config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes, modelID, projectName)

	// Adjust input layer to use CNN first, then FFNN and LSTM
	config.Layers.Input.LayerType = "conv" // Use CNN for input processing

	// Define hidden layers combining FFNN, LSTM, and CNN
	config.Layers.Hidden = []dense.Layer{
		{
			LayerType: "conv",
			Filters: []dense.Filter{
				{
					Weights: dense.Random2DSlice(3, 3),
					Bias:    rand.Float64(),
				},
			},
			Stride:  1,
			Padding: 1,
		},
		{
			LayerType: "dense", // FFNN Layer
			Neurons: func() map[string]dense.Neuron {
				neurons := make(map[string]dense.Neuron)
				for i := 0; i < 128; i++ {
					neuronID := fmt.Sprintf("hidden%d", i)
					neurons[neuronID] = dense.Neuron{
						ActivationType: "relu",
						Bias:           rand.Float64(),
						Connections: func() map[string]dense.Connection {
							connections := make(map[string]dense.Connection)
							for j := 0; j < numInputs; j++ {
								inputID := fmt.Sprintf("input%d", j)
								connections[inputID] = dense.Connection{Weight: rand.NormFloat64()}
							}
							return connections
						}(),
					}
				}
				return neurons
			}(),
		},
		{
			LayerType: "lstm", // LSTM Layer
			LSTMCells: []dense.LSTMCell{ // Use LSTMCells instead of Cells
				{
					InputWeights:  dense.RandomSlice(numInputs),
					ForgetWeights: dense.RandomSlice(numInputs),
					OutputWeights: dense.RandomSlice(numInputs),
					CellWeights:   dense.RandomSlice(numInputs),
					Bias:          rand.Float64(),
				},
			},
		},
	}

	// Adjust output layer connections
	config.Layers.Output.Neurons = make(map[string]dense.Neuron)
	config.Layers.Output.LayerType = "dense"
	for i := 0; i < numOutputs; i++ {
		neuronID := fmt.Sprintf("output%d", i)
		config.Layers.Output.Neurons[neuronID] = dense.Neuron{
			ActivationType: "softmax",
			Bias:           rand.Float64(),
			Connections: func() map[string]dense.Connection {
				connections := make(map[string]dense.Connection)
				for j := 0; j < 128; j++ {
					hiddenID := fmt.Sprintf("hidden%d", j)
					connections[hiddenID] = dense.Connection{Weight: rand.NormFloat64()}
				}
				return connections
			}(),
		}
	}

	return config
}
