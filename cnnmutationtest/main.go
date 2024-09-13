package main

import (
	"dense"
	"fmt"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Test 1: FFNN + CNN Model
	fmt.Println("Test 1: FFNN + CNN Model")

	// Create a model configuration with FFNN and CNN layers
	config1 := dense.CreateRandomNetworkConfig(3, 1, []string{"relu"}, "ffnn_cnn_test", "FFNN + CNN Test")
	addConvLayer(config1)
	addDenseLayer(config1)

	// Apply CNN mutations
	applyCNNMutations(config1)

	// Test 2: CNN + LSTM Model
	fmt.Println("\nTest 2: CNN + LSTM Model")

	// Create a model configuration with CNN and LSTM layers
	config2 := dense.CreateRandomNetworkConfig(3, 1, []string{"relu"}, "cnn_lstm_test", "CNN + LSTM Test")
	addConvLayer(config2)
	addLSTMLayer(config2)

	// Apply CNN mutations
	applyCNNMutations(config2)

	// Test 3: CNN + LSTM + CNN Model
	fmt.Println("\nTest 3: CNN + LSTM + CNN Model")

	// Create a model configuration with CNN, LSTM, and CNN layers
	config3 := dense.CreateRandomNetworkConfig(3, 1, []string{"relu"}, "cnn_lstm_cnn_test", "CNN + LSTM + CNN Test")
	addConvLayer(config3)
	addLSTMLayer(config3)
	addConvLayer(config3)

	// Apply CNN mutations
	applyCNNMutations(config3)
}

// Adds a CNN layer to the config
func addConvLayer(config *dense.NetworkConfig) {
	newLayer := dense.Layer{
		LayerType: "conv",
		Filters: []dense.Filter{
			{
				Weights: dense.Random2DSlice(3, 3),
				Bias:    rand.Float64(),
			},
		},
		Stride:  1,
		Padding: 1,
	}

	config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
}

// Adds an LSTM layer to the config
func addLSTMLayer(config *dense.NetworkConfig) {
	newLayer := dense.Layer{
		LayerType: "lstm",
		LSTMCells: []dense.LSTMCell{
			{
				InputWeights:  dense.RandomSlice(3),
				ForgetWeights: dense.RandomSlice(3),
				OutputWeights: dense.RandomSlice(3),
				CellWeights:   dense.RandomSlice(3),
				Bias:          rand.Float64(),
			},
		},
	}

	config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
}

// Adds a Dense layer to the config
func addDenseLayer(config *dense.NetworkConfig) {
	newLayer := dense.Layer{
		LayerType: "dense",
		Neurons: map[string]dense.Neuron{
			"hidden1": {
				ActivationType: "relu",
				Connections: map[string]dense.Connection{
					"input0": {Weight: rand.Float64()},
					"input1": {Weight: rand.Float64()},
					"input2": {Weight: rand.Float64()},
				},
				Bias: rand.Float64(),
			},
		},
	}

	config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
}

// Apply CNN mutations to the network
func applyCNNMutations(config *dense.NetworkConfig) {
	fmt.Println("Applying MutateCNNWeights...")
	dense.MutateCNNWeights(config, 0.01, 20)

	printCNNLayer(config)

	fmt.Println("Applying MutateCNNBiases...")
	dense.MutateCNNBiases(config, 20, 0.01)

	printCNNLayer(config)

	fmt.Println("Applying RandomizeCNNWeights...")
	dense.RandomizeCNNWeights(config, 20)

	printCNNLayer(config)

	fmt.Println("Applying InvertCNNWeights...")
	dense.InvertCNNWeights(config, 20)

	printCNNLayer(config)

	fmt.Println("Applying AddCNNLayerAtRandomPosition...")
	dense.AddCNNLayerAtRandomPosition(config, 20)

	printCNNLayer(config)
}

// Helper function to print CNN layer details
func printCNNLayer(config *dense.NetworkConfig) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" {
			fmt.Println("CNN Layer:")
			for i, filter := range layer.Filters {
				fmt.Printf("  Filter %d:\n", i)
				fmt.Printf("    Weights:  %v\n", filter.Weights)
				fmt.Printf("    Bias:     %v\n", filter.Bias)
			}
		}
	}
}
