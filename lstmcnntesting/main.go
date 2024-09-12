package main

import (
	"fmt"
	"math/rand"
	"time"

	"dense" // Assuming dense is in the same directory or properly imported
)

func main() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Test 1: FFNN Only
	fmt.Println("Test 1: FFNN Only")
	ffnnConfig := createFFNNModel()
	testFFNN(ffnnConfig)

	// Test 2: FFNN with LSTM
	fmt.Println("\nTest 2: FFNN with LSTM")
	ffnnLSTMConfig := createFFNNLSTMModel()
	testFFNNLSTM(ffnnLSTMConfig)

	// Test 3: FFNN with CNN
	fmt.Println("\nTest 3: FFNN with CNN")
	ffnnCNNConfig := createFFNNCNNModel()
	testFFNNCNN(ffnnCNNConfig)

	// Test 4: FFNN with CNN and LSTM
	fmt.Println("\nTest 4: FFNN with CNN and LSTM")
	fullModelConfig := createFullModel()
	testFullModel(fullModelConfig)
}

// Test 1: Create a simple FFNN model
func createFFNNModel() *dense.NetworkConfig {
	numInputs := 3
	numOutputs := 1
	outputActivationTypes := []string{"sigmoid"}
	modelID := "ffnn_model"
	projectName := "FFNN Test"

	config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes, modelID, projectName)

	// Adjust input layer to be dense
	config.Layers.Input.LayerType = "dense"

	// Define hidden layers (simple dense layer)
	config.Layers.Hidden = []dense.Layer{
		{
			LayerType: "dense",
			Neurons: func() map[string]dense.Neuron {
				neurons := make(map[string]dense.Neuron)
				for i := 0; i < 4; i++ {
					neuronID := fmt.Sprintf("hidden%d", i)
					neurons[neuronID] = dense.Neuron{
						ActivationType: "relu",
						Bias:           rand.Float64(),
						Connections: func() map[string]dense.Connection {
							connections := make(map[string]dense.Connection)
							for j := 0; j < numInputs; j++ {
								inputID := fmt.Sprintf("input%d", j)
								connections[inputID] = dense.Connection{Weight: rand.Float64()}
							}
							return connections
						}(),
					}
				}
				return neurons
			}(),
		},
	}

	// Adjust output layer connections
	config.Layers.Output.Neurons = make(map[string]dense.Neuron)
	config.Layers.Output.LayerType = "dense"
	for i := 0; i < numOutputs; i++ {
		neuronID := fmt.Sprintf("output%d", i)
		config.Layers.Output.Neurons[neuronID] = dense.Neuron{
			ActivationType: "sigmoid",
			Bias:           rand.Float64(),
			Connections: func() map[string]dense.Connection {
				connections := make(map[string]dense.Connection)
				for j := 0; j < 4; j++ {
					hiddenID := fmt.Sprintf("hidden%d", j)
					connections[hiddenID] = dense.Connection{Weight: rand.Float64()}
				}
				return connections
			}(),
		}
	}

	return config
}

func testFFNN(config *dense.NetworkConfig) {
	// Prepare input data
	input := map[string]interface{}{
		"input0": 0.5,
		"input1": 0.3,
		"input2": 0.9,
	}

	// Run feedforward
	outputs := dense.Feedforward(config, input)

	// Print outputs
	fmt.Println("FFNN Output:", outputs)
}

// Test 2: Create an FFNN model with an LSTM layer
func createFFNNLSTMModel() *dense.NetworkConfig {
	numInputs := 3
	numOutputs := 1
	outputActivationTypes := []string{"sigmoid"}
	modelID := "ffnn_lstm_model"
	projectName := "FFNN LSTM Test"

	config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes, modelID, projectName)

	// Adjust input layer to be dense
	config.Layers.Input.LayerType = "dense"

	// Define hidden layers (FFNN + LSTM)
	config.Layers.Hidden = []dense.Layer{
		{
			LayerType: "dense",
			Neurons: func() map[string]dense.Neuron {
				neurons := make(map[string]dense.Neuron)
				for i := 0; i < 4; i++ {
					neuronID := fmt.Sprintf("hidden%d", i)
					neurons[neuronID] = dense.Neuron{
						ActivationType: "relu",
						Bias:           rand.Float64(),
						Connections: func() map[string]dense.Connection {
							connections := make(map[string]dense.Connection)
							for j := 0; j < numInputs; j++ {
								inputID := fmt.Sprintf("input%d", j)
								connections[inputID] = dense.Connection{Weight: rand.Float64()}
							}
							return connections
						}(),
					}
				}
				return neurons
			}(),
		},
		{
			LayerType: "lstm",
			LSTMCells: []dense.LSTMCell{
				{
					InputWeights:  randomSlice(4),
					ForgetWeights: randomSlice(4),
					OutputWeights: randomSlice(4),
					CellWeights:   randomSlice(4),
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
			ActivationType: "sigmoid",
			Bias:           rand.Float64(),
			Connections: func() map[string]dense.Connection {
				connections := make(map[string]dense.Connection)
				connections["lstm0"] = dense.Connection{Weight: rand.Float64()}
				return connections
			}(),
		}
	}

	return config
}

func testFFNNLSTM(config *dense.NetworkConfig) {
	// Prepare input data
	input := map[string]interface{}{
		"input0": 0.5,
		"input1": 0.3,
		"input2": 0.9,
	}

	// Run feedforward
	outputs := dense.Feedforward(config, input)

	// Print outputs
	fmt.Println("FFNN with LSTM Output:", outputs)
}

// Test 3: Create an FFNN model with a CNN layer
func createFFNNCNNModel() *dense.NetworkConfig {
	numInputs := 3
	numOutputs := 1
	outputActivationTypes := []string{"sigmoid"}
	modelID := "ffnn_cnn_model"
	projectName := "FFNN CNN Test"

	config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes, modelID, projectName)

	// Adjust input layer to be convolutional
	config.Layers.Input = dense.Layer{
		LayerType: "conv",
	}

	// Define hidden layers (CNN + FFNN)
	config.Layers.Hidden = []dense.Layer{
		{
			LayerType: "conv",
			Filters: []dense.Filter{
				{
					Weights: [][]float64{
						{rand.Float64(), rand.Float64(), rand.Float64()},
						{rand.Float64(), rand.Float64(), rand.Float64()},
						{rand.Float64(), rand.Float64(), rand.Float64()},
					},
					Bias: rand.Float64(),
				},
			},
			Stride:  1,
			Padding: 1,
		},
		{
			LayerType: "dense",
			Neurons: func() map[string]dense.Neuron {
				neurons := make(map[string]dense.Neuron)
				// Assuming the output from the conv layer is flattened into 9 inputs
				for i := 0; i < 4; i++ {
					neuronID := fmt.Sprintf("hidden%d", i)
					neurons[neuronID] = dense.Neuron{
						ActivationType: "relu",
						Bias:           rand.Float64(),
						Connections: func() map[string]dense.Connection {
							connections := make(map[string]dense.Connection)
							for j := 0; j < 9; j++ {
								inputID := fmt.Sprintf("conv_output%d", j)
								connections[inputID] = dense.Connection{Weight: rand.Float64()}
							}
							return connections
						}(),
					}
				}
				return neurons
			}(),
		},
	}

	// Adjust output layer connections
	config.Layers.Output.Neurons = make(map[string]dense.Neuron)
	config.Layers.Output.LayerType = "dense"
	for i := 0; i < numOutputs; i++ {
		neuronID := fmt.Sprintf("output%d", i)
		config.Layers.Output.Neurons[neuronID] = dense.Neuron{
			ActivationType: "sigmoid",
			Bias:           rand.Float64(),
			Connections: func() map[string]dense.Connection {
				connections := make(map[string]dense.Connection)
				for j := 0; j < 4; j++ {
					hiddenID := fmt.Sprintf("hidden%d", j)
					connections[hiddenID] = dense.Connection{Weight: rand.Float64()}
				}
				return connections
			}(),
		}
	}

	return config
}

func testFFNNCNN(config *dense.NetworkConfig) {
	// Prepare input data (simple 3x3 image)
	image := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}

	input := map[string]interface{}{
		"image": image,
	}

	// Run feedforward
	outputs := dense.Feedforward(config, input)

	// Print outputs
	fmt.Println("FFNN with CNN Output:", outputs)
}

// Test 4: Create a model with FFNN, CNN, and LSTM layers
func createFullModel() *dense.NetworkConfig {
	numInputs := 3
	numOutputs := 1
	outputActivationTypes := []string{"sigmoid"}
	modelID := "full_model"
	projectName := "Full Model Test"

	config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, outputActivationTypes, modelID, projectName)

	// Adjust input layer to be convolutional
	config.Layers.Input = dense.Layer{
		LayerType: "conv",
	}

	// Define hidden layers (CNN + LSTM + FFNN)
	config.Layers.Hidden = []dense.Layer{
		{
			LayerType: "conv",
			Filters: []dense.Filter{
				{
					Weights: [][]float64{
						{rand.Float64(), rand.Float64(), rand.Float64()},
						{rand.Float64(), rand.Float64(), rand.Float64()},
						{rand.Float64(), rand.Float64(), rand.Float64()},
					},
					Bias: rand.Float64(),
				},
			},
			Stride:  1,
			Padding: 1,
		},
		{
			LayerType: "lstm",
			LSTMCells: []dense.LSTMCell{
				{
					InputWeights:  randomSlice(9), // Assuming output from CNN is flattened to length 9
					ForgetWeights: randomSlice(9),
					OutputWeights: randomSlice(9),
					CellWeights:   randomSlice(9),
					Bias:          rand.Float64(),
				},
			},
		},
		{
			LayerType: "dense",
			Neurons: func() map[string]dense.Neuron {
				neurons := make(map[string]dense.Neuron)
				for i := 0; i < 4; i++ {
					neuronID := fmt.Sprintf("hidden%d", i)
					neurons[neuronID] = dense.Neuron{
						ActivationType: "relu",
						Bias:           rand.Float64(),
						Connections: func() map[string]dense.Connection {
							connections := make(map[string]dense.Connection)
							connections["lstm0"] = dense.Connection{Weight: rand.Float64()}
							return connections
						}(),
					}
				}
				return neurons
			}(),
		},
	}

	// Adjust output layer connections
	config.Layers.Output.Neurons = make(map[string]dense.Neuron)
	config.Layers.Output.LayerType = "dense"
	for i := 0; i < numOutputs; i++ {
		neuronID := fmt.Sprintf("output%d", i)
		config.Layers.Output.Neurons[neuronID] = dense.Neuron{
			ActivationType: "sigmoid",
			Bias:           rand.Float64(),
			Connections: func() map[string]dense.Connection {
				connections := make(map[string]dense.Connection)
				for j := 0; j < 4; j++ {
					hiddenID := fmt.Sprintf("hidden%d", j)
					connections[hiddenID] = dense.Connection{Weight: rand.Float64()}
				}
				return connections
			}(),
		}
	}

	return config
}

func testFullModel(config *dense.NetworkConfig) {
	// Prepare input data (simple 3x3 image)
	image := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}

	input := map[string]interface{}{
		"image": image,
	}

	// Run feedforward
	outputs := dense.Feedforward(config, input)

	// Print outputs
	fmt.Println("Full Model Output:", outputs)
}

// Helper function to create random slice of floats
func randomSlice(length int) []float64 {
	slice := make([]float64, length)
	for i := range slice {
		slice[i] = rand.Float64()
	}
	return slice
}
