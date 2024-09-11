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

// Optimization process that runs hill climbing with batching and multithreading
func hillClimbingOptimize(mnist *dense.MNISTData, iterations, batchSize, numWorkers int, learningRate, fitnessBuffer float64) float64 {
    // Load the best config from the file only once
    bestConfig, err := dense.LoadNetworkFromFile("best_model.json")
    if err != nil {
        fmt.Println("Error loading best model, using default config:", err)
        return 0.0
    }

    // Split data into training and testing sets
    trainData, testData := splitData(mnist)

    // Evaluate initial accuracy on the training set
    bestFitness := evaluateFitness(bestConfig, trainData)
    fmt.Printf("Starting optimization with initial accuracy (training set): %.4f%%\n", bestFitness*100)

    for i := 0; i < iterations; i += batchSize {
        var wg sync.WaitGroup
        results := make(chan Result, batchSize)

        // Start a batch of workers
        for j := 0; j < batchSize; j++ {
            wg.Add(1)
            go func(workerID int) {
                defer wg.Done()
                tmpModelFilename := fmt.Sprintf("tmp_model_%d.json", workerID)

                // Load the temporary model file for this worker
                currentConfig, err := dense.LoadNetworkFromFile("best_model.json")
                if err != nil {
                    fmt.Println("Error loading best model during iteration:", err)
                    return
                }

                // Mutate the model and evaluate its fitness
                dense.MutateNetwork(currentConfig, learningRate, 30)
                newFitness := evaluateFitness(currentConfig, trainData) // Evaluate on training data

                // Save mutated model to the temporary file again
                dense.SaveNetworkToFile(currentConfig, tmpModelFilename)

                results <- Result{fitness: newFitness, tmpModel: tmpModelFilename}
            }(j % numWorkers) // Distribute jobs across workers
        }

        // Wait for the batch to complete
        go func() {
            wg.Wait()
            close(results) // Only close the channel after all Goroutines are done
        }()

        // After all threads finish in the batch, evaluate the results
        for result := range results {
            if result.fitness > bestFitness+fitnessBuffer {
                bestFitness = result.fitness
                // Load the best temp model and save it as the new best model
                tmpConfig, err := dense.LoadNetworkFromFile(result.tmpModel)
                if err != nil {
                    fmt.Println("Error loading temp model:", err)
                    continue
                }
                dense.SaveNetworkToFile(tmpConfig, "best_model.json") // Overwrite the best model
                fmt.Printf("New best model found with accuracy: %.4f%%\n", bestFitness*100)
            }

            // Delete the temporary model file after it's no longer needed
            err = os.Remove(result.tmpModel)
            if err != nil {
                fmt.Printf("Warning: Failed to delete temp model file: %s\n", result.tmpModel)
            }
        }

        fmt.Printf("\nBatch ending at iteration %d: Current best accuracy: %.4f%%\n", i+batchSize, bestFitness*100)
    }

    // After hill climbing, evaluate on the test data
    finalConfig, err := dense.LoadNetworkFromFile("best_model.json")
    if err == nil {
        testAccuracy := evaluateFitness(finalConfig, testData) // Evaluate on test data
        fmt.Printf("Final model accuracy on test set: %.4f%%\n", testAccuracy*100)
    }

    return bestFitness * 100
}

func modelExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	err := dense.EnsureMNISTDownloads()
	if err != nil {
		fmt.Println("Error downloading MNIST dataset:", err)
		return
	}

	mnist, err := dense.LoadMNIST()
	if err != nil {
		fmt.Println("Error loading MNIST dataset:", err)
		return
	}

	// Check if the best model already exists
	if !modelExists("best_model.json") {
		fmt.Println("No existing model found, creating a new one...")
		numInputs := 28 * 28
		numOutputs := 1
		outputActivationTypes := []string{"sigmoid"}
		initialConfig := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes)
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
		bestFitness := hillClimbingOptimize(mnist, 100, batchSize, numWorkers, 0.1, 0.0001)
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
