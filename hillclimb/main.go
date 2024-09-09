package main

import (
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"time"

	"dense"
)

// Result struct to store the results from each worker (goroutine)
type Result struct {
	fitness   float64
	iteration int
	config    *dense.NetworkConfig
}

// EvaluateFitness will now train the network using the MNIST data and calculate accuracy
func evaluateFitness(config *dense.NetworkConfig, mnist *dense.MNISTData, rng *rand.Rand) float64 {
	correct := 0

	for i, image := range mnist.Images {
		// Convert the image to a format the network can process
		input := make(map[string]float64)
		for j, pixel := range image {
			input[fmt.Sprintf("input%d", j)] = float64(pixel) / 255.0 // Normalize pixel values
		}

		// Get the expected label
		expectedLabel := mnist.Labels[i]

		// Get the network's prediction
		outputs := dense.Feedforward(config, input)

		// Find the output neuron with the highest value (this is the network's prediction)
		maxOutput := -1.0
		predictedLabel := -1
		for idxStr, output := range outputs {
			idx, _ := strconv.Atoi(idxStr) // Convert string key to int
			if output > maxOutput {
				maxOutput = output
				predictedLabel = idx
			}
		}

		// Compare predicted label with actual label
		if predictedLabel == int(expectedLabel) {
			correct++
		}
	}

	// Calculate accuracy
	accuracy := float64(correct) / float64(len(mnist.Images))
	return accuracy
}

func hillClimbingOptimize(config *dense.NetworkConfig, mnist *dense.MNISTData, iterations int, learningRate float64) {
    bestFitness := evaluateFitness(config, mnist, rand.New(rand.NewSource(time.Now().UnixNano())))
    bestConfig := config

    batchSize := 5  // How many iterations to do in a single batch
    for i := 0; i < iterations; i += batchSize {

        var wg sync.WaitGroup
        results := make(chan Result, batchSize)

        // Run a batch of workers
        for j := 0; j < batchSize; j++ {
            wg.Add(1)
            go func(iteration int) {
                defer wg.Done()

                // Create an independent RNG for each worker
                rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(iteration)))

                config.mutex.RLock()  // Lock for reading before copying config
                currentConfig := *bestConfig
                config.mutex.RUnlock()  // Unlock after reading

                // Apply a mutation to the copied config
                dense.MutateNetwork(&currentConfig, learningRate, 30)

                // Evaluate the new configuration's fitness
                newFitness := evaluateFitness(&currentConfig, mnist, rng)

                results <- Result{
                    fitness:   newFitness,
                    iteration: iteration,
                    config:    &currentConfig,
                }
            }(i + j)
        }

        // Wait for all workers in this batch to finish
        wg.Wait()
        close(results)

        // Evaluate results after batch
        for result := range results {
            if result.fitness > bestFitness {
                bestFitness = result.fitness
                config.mutex.Lock()  // Lock for writing before updating the bestConfig
                bestConfig = result.config
                config.mutex.Unlock()  // Unlock after writing

                fmt.Printf("Iteration %d: New best model with accuracy %.4f%%\n", result.iteration, bestFitness*100)
                dense.SaveNetworkToFile(bestConfig, "best_model.json")
            }
        }

        fmt.Printf("Batch ending at iteration %d: Current best accuracy %.4f%%\n", i+batchSize, bestFitness*100)
    }

    // Final best model save
    dense.SaveNetworkToFile(bestConfig, "final_best_model.json")
    fmt.Printf("Final best model saved with accuracy: %.4f%%\n", bestFitness*100)
}


func main() {
	rand.Seed(time.Now().UnixNano())

	// Ensure MNIST dataset is downloaded
	err := dense.EnsureMNISTDownloads()
	if err != nil {
		fmt.Println("Error downloading MNIST dataset:", err)
		return
	}

	// Load MNIST dataset
	mnist, err := dense.LoadMNIST()
	if err != nil {
		fmt.Println("Error loading MNIST dataset:", err)
		return
	}

	// Create a random network configuration
	config := dense.CreateRandomNetworkConfig()

	fmt.Println("Starting hill climbing optimization...")

	// Run hill climbing optimization
	hillClimbingOptimize(config, mnist, 1000, 0.1)

	// Load and print the best saved model
	loadedConfig, err := dense.LoadNetworkFromFile("final_best_model.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}
	fmt.Println("Loaded best model configuration:", loadedConfig)
}
