package main

import (
	"fmt"
	"math/rand"
	//"strconv"
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

		// Check if the image size is correct
		if len(image) != 28*28 {
			fmt.Printf("Error: Image %d does not have the correct dimensions. Got %d pixels, expected 784.\n", i, len(image))
			continue
		}

		// Normalize pixel values and prepare the input
		for j, pixel := range image {
			input[fmt.Sprintf("input%d", j)] = float64(pixel) / 255.0 // Normalize pixel values
		}

		// Debug: Print the input data for the first few images to ensure it's being loaded correctly
		/*if i < 3 { // Print only for the first 3 images for clarity
			fmt.Printf("Image %d input data: %v\n", i, input)
		}*/

		// Get the expected label
		expectedLabel := mnist.Labels[i]

		// Debug: Print the expected label for the first few images
		/*if i < 3 {
			fmt.Printf("Image %d expected label: %d\n", i, expectedLabel)
		}*/

		// Get the network's prediction
		outputs := dense.Feedforward(config, input)

		// Since we now only have one output, it will predict the label as a single number from 0 to 9
		predictedLabel := int(outputs["output0"] * 9.0) // Scale the output to get a number between 0 and 9

		// Debug: Print the predicted label and the outputs for the first few images
		/*if i < 3 {
			fmt.Printf("Image %d predicted label: %d, output: %v\n", i, predictedLabel, outputs)
		}*/

		// Compare predicted label with actual label
		if predictedLabel == int(expectedLabel) {
			correct++
		}
	}

	// Calculate accuracy
	accuracy := float64(correct) / float64(len(mnist.Images))

	// Debug: Print the final accuracy
	fmt.Printf("Final accuracy: %.4f%%\n", accuracy*100)

	return accuracy
}


func hillClimbingOptimize(config *dense.NetworkConfig, mnist *dense.MNISTData, iterations int, learningRate float64) {
    bestFitness := evaluateFitness(config, mnist, rand.New(rand.NewSource(time.Now().UnixNano())))
    bestConfig := config

    batchSize := 10 // How many iterations to do in a single batch
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

                // Create a deep copy of the best configuration for this thread
                currentConfig := dense.DeepCopy(bestConfig)

                // Apply a mutation to the copied config
                dense.MutateNetwork(currentConfig, learningRate, 30)

                // Evaluate the new configuration's fitness
                newFitness := evaluateFitness(currentConfig, mnist, rng)

                // Send the result to the channel
                results <- Result{
                    fitness:   newFitness,
                    iteration: iteration,
                    config:    currentConfig,
                }
            }(i + j)
        }

        // Wait for all workers in this batch to finish
        wg.Wait()
        close(results)

        // Evaluate results after batch and update the best model if a better one is found
        for result := range results {
            if result.fitness > bestFitness {
                bestFitness = result.fitness
                bestConfig = result.config // Use the best configuration found in this batch

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

	// Set dynamic input and output sizes based on the dataset
	numInputs := 28 * 28  // MNIST images are 28x28 pixels, so 784 input neurons
	numOutputs := 1       // Single output neuron for predicting digits 0 to 9

	// Custom activation function for output neuron
	outputActivationTypes := []string{"sigmoid"} // Use sigmoid activation for output

	// Create the network configuration
	config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes)

	fmt.Println("Starting hill climbing optimization...")

	// Run hill climbing optimization
	hillClimbingOptimize(config, mnist, 500, 0.1)

	// Load and print the best saved model
	loadedConfig, err := dense.LoadNetworkFromFile("final_best_model.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}
	fmt.Println("Loaded best model configuration:", loadedConfig)
}
