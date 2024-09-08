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

// Mutex for safe printing
var printMutex sync.Mutex

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
	numWorkers := 4
	results := make(chan Result, numWorkers)
	var wg sync.WaitGroup

	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(iteration int) {
			defer wg.Done()

			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(iteration))) // Independent RNG per goroutine
			currentConfig := *bestConfig
			dense.MutateWeights(&currentConfig, learningRate)
			newFitness := evaluateFitness(&currentConfig, mnist, rng)

			results <- Result{
				fitness:   newFitness,
				iteration: iteration,
				config:    &currentConfig,
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	for result := range results {
		if result.fitness > bestFitness {
			bestFitness = result.fitness
			bestConfig = result.config
			printMutex.Lock() // Lock console output to avoid race conditions
			fmt.Printf("Iteration %d: Found better model with accuracy %.4f\n", result.iteration, bestFitness*100)
			dense.SaveNetworkToFile(bestConfig, fmt.Sprintf("model_iteration_%d.json", result.iteration))
			printMutex.Unlock()
		}
	}

	dense.SaveNetworkToFile(bestConfig, "best_model.json")
	printMutex.Lock()
	fmt.Println("Best model saved to best_model.json")
	printMutex.Unlock()
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
	hillClimbingOptimize(config, mnist, 100, 0.05)

	// Load and print the best saved model
	loadedConfig, err := dense.LoadNetworkFromFile("best_model.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}
	fmt.Println("Loaded best model configuration:", loadedConfig)
}
