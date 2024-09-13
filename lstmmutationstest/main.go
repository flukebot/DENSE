package main

import (
	"fmt"
	"math/rand"
	"time"

	"dense" // Ensure the "dense" import path is correct for your project
)

func main() {
    // Seed the random number generator
    rand.Seed(time.Now().UnixNano())

    // Create models for testing
    fmt.Println("Test 1: LSTM Only Model")
    lstmOnlyModel := createLSTMOnlyModel()
    testMutations(lstmOnlyModel)

    fmt.Println("\nTest 2: LSTM + FFNN Model")
    lstmFFNNModel := createLSTMFFNNModel()
    testMutations(lstmFFNNModel)

    fmt.Println("\nTest 3: LSTM + FFNN + CNN Model")
    lstmFFNNCNNModel := createLSTMFFNNCNNModel()
    testMutations(lstmFFNNCNNModel)
}

// Create an LSTM-only model
func createLSTMOnlyModel() *dense.NetworkConfig {
    modelID := "lstm_only_model"
    projectName := "LSTM Only Test"
    numInputs := 3
    numOutputs := 1

    config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, []string{"sigmoid"}, modelID, projectName)

    // Define input layer (Dense input to LSTM)
    config.Layers.Input.LayerType = "dense"

    // Define LSTM hidden layer
    config.Layers.Hidden = []dense.Layer{
        {
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
        },
    }

    return config
}

// Create a model with LSTM and FFNN layers
func createLSTMFFNNModel() *dense.NetworkConfig {
    modelID := "lstm_ffnn_model"
    projectName := "LSTM FFNN Test"
    numInputs := 3
    numOutputs := 1

    config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, []string{"sigmoid"}, modelID, projectName)

    // Define input layer (Dense input to LSTM)
    config.Layers.Input.LayerType = "dense"

    // Define LSTM + FFNN hidden layers
    config.Layers.Hidden = []dense.Layer{
        {
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
        },
        {
            LayerType: "dense",
            Neurons: map[string]dense.Neuron{
                "hidden1": {
                    ActivationType: "relu",
                    Bias:           rand.Float64(),
                    Connections: func() map[string]dense.Connection {
                        connections := make(map[string]dense.Connection)
                        connections["lstm_output"] = dense.Connection{Weight: rand.Float64()}
                        return connections
                    }(),
                },
            },
        },
    }

    return config
}

// Create a model with LSTM, FFNN, and CNN layers
func createLSTMFFNNCNNModel() *dense.NetworkConfig {
    modelID := "lstm_ffnn_cnn_model"
    projectName := "LSTM FFNN CNN Test"
    numInputs := 3
    numOutputs := 1

    config := dense.CreateRandomNetworkConfig(numInputs, numOutputs, []string{"sigmoid"}, modelID, projectName)

    // Define input layer (Dense input to LSTM)
    config.Layers.Input.LayerType = "dense"

    // Define LSTM + FFNN + CNN hidden layers
    config.Layers.Hidden = []dense.Layer{
        {
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
        },
        {
            LayerType: "dense",
            Neurons: map[string]dense.Neuron{
                "hidden1": {
                    ActivationType: "relu",
                    Bias:           rand.Float64(),
                    Connections: func() map[string]dense.Connection {
                        connections := make(map[string]dense.Connection)
                        connections["lstm_output"] = dense.Connection{Weight: rand.Float64()}
                        return connections
                    }(),
                },
            },
        },
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
    }

    return config
}

// Test all mutations for LSTM layers
func testMutations(config *dense.NetworkConfig) {
    fmt.Println("\nApplying MutateLSTMWeights...")
    dense.MutateLSTMWeights(config, 0.01, 50)
    printLSTMDetails(config)

    fmt.Println("\nApplying MutateLSTMBiases...")
    dense.MutateLSTMBiases(config, 50, 0.01)
    printLSTMDetails(config)

    fmt.Println("\nApplying RandomizeLSTMWeights...")
    dense.RandomizeLSTMWeights(config, 50)
    printLSTMDetails(config)

    fmt.Println("\nApplying InvertLSTMWeights...")
    dense.InvertLSTMWeights(config, 50)
    printLSTMDetails(config)

    fmt.Println("\nApplying AddLSTMLayerAtRandomPosition...")
    dense.AddLSTMLayerAtRandomPosition(config, 50)
    printLSTMDetails(config)
}

// Helper function to print LSTM layer details for verification
func printLSTMDetails(config *dense.NetworkConfig) {
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "lstm" {
            fmt.Println("LSTM Layer:")
            for i, cell := range layer.LSTMCells {
                fmt.Printf("  Cell %d: \n", i)
                fmt.Printf("    InputWeights:  %v\n", cell.InputWeights)
                fmt.Printf("    ForgetWeights: %v\n", cell.ForgetWeights)
                fmt.Printf("    OutputWeights: %v\n", cell.OutputWeights)
                fmt.Printf("    CellWeights:   %v\n", cell.CellWeights)
                fmt.Printf("    Bias:          %v\n", cell.Bias)
            }
        }
    }
}
