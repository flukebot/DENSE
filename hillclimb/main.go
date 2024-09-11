package main

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"dense" // Assuming this is your custom package for the neural network
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
		Labels: mnist.Labels[:splitIndex],
	}

	testData = &dense.MNISTData{
		Images: mnist.Images[splitIndex:],
		Labels: mnist.Labels[splitIndex:],
	}

	return trainData, testData
}

// Evaluates the fitness of the model using the provided dataset
func evaluateFitness(config *dense.NetworkConfig, mnist *dense.MNISTData) float64 {
	correct := 0
	total := len(mnist.Images)
	for i, image := range mnist.Images {
		input := make(map[string]float64)
		for j, pixel := range image {
			input[fmt.Sprintf("input%d", j)] = float64(pixel) / 255.0
		}
		expectedLabel := mnist.Labels[i]
		outputs := dense.Feedforward(config, input)
		predictedLabel := int(outputs["output0"] * 9.0)
		if predictedLabel == int(expectedLabel) {
			correct++
		}
	}
	accuracy := float64(correct) / float64(total)
	return accuracy
}

func hillClimbingOptimize(trainData, testData *dense.MNISTData, iterations, batchSize, numWorkers int, learningRate, fitnessBuffer float64) float64 {
    // Load the best config from the file only once
    bestConfig, err := dense.LoadNetworkFromFile("best_model.json")
    if err != nil {
        fmt.Println("Error loading best model, using default config:", err)
        return 0.0
    }

    // Load initial accuracies from metadata
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

        // After all threads finish in the batch, evaluate the results
        for result := range results {
            if result.fitness > bestTrainingAccuracy+fitnessBuffer {
                bestTrainingAccuracy = result.fitness
                // Load the best temp model and store it in memory (don't save it to file just yet)
                tmpConfig, err := dense.LoadNetworkFromFile(result.tmpModel)
                if err != nil {
                    fmt.Println("Error loading temp model:", err)
                    continue
                }
                bestConfig = tmpConfig // Update the best config in memory
                fmt.Printf("New best model found with accuracy: %.4f%%\n", bestTrainingAccuracy*100)
            }

            // Delete the temporary model file after it's no longer needed
            err = os.Remove(result.tmpModel)
            if err != nil {
                fmt.Printf("Warning: Failed to delete temp model file: %s\n", result.tmpModel)
            }
        }

        fmt.Printf("\nBatch ending at iteration %d: Current best training accuracy: %.4f%%\n", i+batchSize, bestTrainingAccuracy*100)
    }

    // Update metadata with new evaluations
    bestConfig.Metadata.LastTrainingAccuracy = bestTrainingAccuracy
    bestConfig.Metadata.LastTestAccuracy = evaluateFitness(bestConfig, testData) // Evaluate on test data
    fmt.Printf("Final model accuracy on test set: %.4f%%\n", bestConfig.Metadata.LastTestAccuracy*100)

    // Save the best model after all batches are complete
    dense.SaveNetworkToFile(bestConfig, "best_model.json")

    return bestTrainingAccuracy * 100
}


func modelExists(filename string) bool {
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

	err := dense.EnsureMNISTDownloads()
	if err != nil {
		fmt.Println("Error downloading MNIST dataset:", err)
		return
	}

	var trainData, testData *dense.MNISTData
	trainFile := "train_data.json"
	testFile := "test_data.json"

	// Check if the training and test data already exist
	if modelExists(trainFile) && modelExists(testFile) {
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
	if !modelExists("best_model.json") {
		fmt.Println("No existing model found, creating a new one...")
		numInputs := 28 * 28
		numOutputs := 1
		outputActivationTypes := []string{"sigmoid"}
		initialConfig := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes, "model1", "project1")
		dense.SaveNetworkToFile(initialConfig, "best_model.json") // Save the initial model
	} else {
		fmt.Println("Loading existing best model from file...")
	}

	fmt.Println("Starting multithreaded hill climbing optimization...")

	desiredAccuracy := 80.0
	maxDuration := 12 * time.Hour
	startTime := time.Now()

	numWorkers := 10 // Adjust this based on your system's capabilities
	batchSize := 10  // Set the batch size

	for {
		bestFitness := hillClimbingOptimize(trainData, testData, 100, batchSize, numWorkers, 0.5, 0.005)
		fmt.Printf("Current best model accuracy (training set): %.4f%%\n", bestFitness)

		if bestFitness >= desiredAccuracy || time.Since(startTime) >= maxDuration {
			break
		}

		time.Sleep(1 * time.Second)
	}

	// Load and print the best saved model
	loadedConfig, err := dense.LoadNetworkFromFile("best_model.json") // Load the final best model
	if err != nil {
		fmt.Println("Error loading final best model:", err)
		return
	}
	fmt.Println("Loaded best model configuration:", loadedConfig)
}
