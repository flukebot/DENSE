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
	fitness float64
	config  *dense.NetworkConfig
}

func evaluateFitness(config *dense.NetworkConfig, mnist *dense.MNISTData, rng *rand.Rand) float64 {
	correct := 0
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
	accuracy := float64(correct) / float64(len(mnist.Images))
	return accuracy
}

func hillClimbingOptimize(config *dense.NetworkConfig, mnist *dense.MNISTData, iterations, numWorkers int, learningRate, fitnessBuffer float64) float64 {
	bestFitness := evaluateFitness(config, mnist, rand.New(rand.NewSource(time.Now().UnixNano())))
	bestConfig := dense.DeepCopy(config)

	fmt.Printf("Starting optimization with initial accuracy: %.4f%%\n", bestFitness*100)

	var wg sync.WaitGroup
	results := make(chan Result, iterations)

	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			currentConfig := dense.DeepCopy(bestConfig)
			dense.MutateNetwork(currentConfig, learningRate, 30)
			newFitness := evaluateFitness(currentConfig, mnist, rand.New(rand.NewSource(time.Now().UnixNano())))
			results <- Result{fitness: newFitness, config: currentConfig}
		}()

		if (i+1)%numWorkers == 0 || i == iterations-1 {
			wg.Wait()
			close(results)

			for result := range results {
				if result.fitness > bestFitness+fitnessBuffer {
					bestFitness = result.fitness
					bestConfig = dense.DeepCopy(result.config)
					fmt.Printf("Iteration %d: New best model with accuracy %.4f%%\n", i, bestFitness*100)
					dense.SaveNetworkToFile(bestConfig, "best_model.json")
				}
			}

			if (i+1)%10 == 0 {
				fmt.Printf("\nCurrent best accuracy: %.4f%%\n", bestFitness*100)
			}

			results = make(chan Result, iterations)
		}
	}

	dense.SaveNetworkToFile(bestConfig, "final_best_model.json")
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

	var config *dense.NetworkConfig
	if modelExists("final_best_model.json") {
		fmt.Println("Loading existing model from file...")
		config, err = dense.LoadNetworkFromFile("final_best_model.json")
		if err != nil {
			fmt.Println("Error loading model:", err)
			return
		}
	} else {
		fmt.Println("No existing model found, creating a new one...")
		numInputs := 28 * 28
		numOutputs := 1
		outputActivationTypes := []string{"sigmoid"}
		config = dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes)
	}

	fmt.Println("Starting multithreaded hill climbing optimization...")

	desiredAccuracy := 60.0
	maxDuration := 12 * time.Hour
	startTime := time.Now()
	numWorkers := 10 // Adjust this based on your system's capabilities

	for {
		bestFitness := hillClimbingOptimize(config, mnist, 100, numWorkers, 0.1, 0.0001)
		fmt.Printf("Current best model accuracy: %.4f%%\n", bestFitness)

		if bestFitness >= desiredAccuracy || time.Since(startTime) >= maxDuration {
			break
		}

		time.Sleep(1 * time.Second)
	}

	loadedConfig, err := dense.LoadNetworkFromFile("final_best_model.json")
	if err != nil {
		fmt.Println("Error loading final model:", err)
		return
	}
	fmt.Println("Loaded best model configuration:", loadedConfig)
}