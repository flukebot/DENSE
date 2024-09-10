package main

import (
	"fmt"
	"math/rand"
	"os"
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
		input := make(map[string]float64)

		if len(image) != 28*28 {
			fmt.Printf("Error: Image %d does not have the correct dimensions. Got %d pixels, expected 784.\n", i, len(image))
			continue
		}

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

	accuracy := float64(correct) / float64(len(mnist.Images))
	fmt.Printf("Final accuracy: %.4f%%\n", accuracy*100)
	return accuracy
}

func hillClimbingOptimize(config *dense.NetworkConfig, mnist *dense.MNISTData, iterations int, learningRate float64) float64{
	bestFitness := evaluateFitness(config, mnist, rand.New(rand.NewSource(time.Now().UnixNano())))
	bestConfig := config

	batchSize := 10
	for i := 0; i < iterations; i += batchSize {

		var wg sync.WaitGroup
		results := make(chan Result, batchSize)

		for j := 0; j < batchSize; j++ {
			wg.Add(1)
			go func(iteration int) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(iteration)))
				currentConfig := dense.DeepCopy(bestConfig)
				dense.MutateNetwork(currentConfig, learningRate, 30)
				newFitness := evaluateFitness(currentConfig, mnist, rng)

				results <- Result{
					fitness:   newFitness,
					iteration: iteration,
					config:    currentConfig,
				}
			}(i + j)
		}

		wg.Wait()
		close(results)

		for result := range results {
			if result.fitness > bestFitness {
				bestFitness = result.fitness
				bestConfig = result.config
				fmt.Printf("Iteration %d: New best model with accuracy %.4f%%\n", result.iteration, bestFitness*100)
				dense.SaveNetworkToFile(bestConfig, "best_model.json")
			}
		}

		fmt.Printf("Batch ending at iteration %d: Current best accuracy %.4f%%\n", i+batchSize, bestFitness*100)
	}

	dense.SaveNetworkToFile(bestConfig, "final_best_model.json")
	//fmt.Printf("Final best model saved with accuracy: %.4f%%\n", bestFitness*100)
	return bestFitness*100
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
	var config *dense.NetworkConfig
	if modelExists("final_best_model.json") {
		fmt.Println("Loading existing model from file...")
		config, err = dense.LoadNetworkFromFile("final_best_model.json")
		if err != nil {
			fmt.Println("Error loading model:", err)
			return
		}
	} else {
		// Set up a new random network if no model exists
		fmt.Println("No existing model found, creating a new one...")
		numInputs := 28 * 28
		numOutputs := 1
		outputActivationTypes := []string{"sigmoid"}
		config = dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes)
	}

	fmt.Println("Starting hill climbing optimization...")


	desiredAccuracy := 60.0
	maxDuration := 12 * time.Hour

	startTime := time.Now()
	var bestFitness float64

	for {
		bestFitness = hillClimbingOptimize(config, mnist, 10, 0.1)
		fmt.Printf("Current best model accuracy: %.4f%%\n", bestFitness*100)

		// Check if the desired accuracy is reached or if the time limit is exceeded
		if bestFitness >= desiredAccuracy || time.Since(startTime) >= maxDuration {
			break
		}

		// Sleep for a short duration to avoid tight looping (optional)
		time.Sleep(1 * time.Second)
	}


	// Run hill climbing optimization
	//bestFitness := hillClimbingOptimize(config, mnist, 10, 0.1)
	//fmt.Printf("Final best model saved with accuracy: %.4f%%\n", bestFitness)

	// Load and print the best saved model
	loadedConfig, err := dense.LoadNetworkFromFile("final_best_model.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
		return
	}
	fmt.Println("Loaded best model configuration:", loadedConfig)
}
