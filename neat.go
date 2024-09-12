package dense

import (
	"dense" // Assuming dense.go is in the same package or imported correctly
	"math/rand"
	"strconv"
)

// CreateSimpleNeuralNetworkWithParams creates a neural network with 3 layers and takes input, output sizes, and metadata as parameters.
func CreateSimpleNeuralNetworkWithParams(
	numInputs int,
	numOutputs int,
	outputActivationTypes []string,
	modelID string,
	projectName string,
) *dense.NetworkConfig {
	// Initialize the metadata
	network := &dense.NetworkConfig{
		Metadata: dense.ModelMetadata{
			ModelID:            modelID,
			ProjectName:        projectName,
			LastTrainingAccuracy: 0.0,
			LastTestAccuracy:     0.0,
		},
	}

	// Initialize input neurons based on the passed number of input neurons
	network.Layers.Input.Neurons = make(map[string]dense.Neuron)
	for i := 0; i < numInputs; i++ {
		neuronID := "input" + strconv.Itoa(i)
		network.Layers.Input.Neurons[neuronID] = dense.Neuron{}
	}

	// Initialize hidden layer with 3 neurons
	numHiddenNeurons := 3
	hiddenLayer := dense.Layer{Neurons: make(map[string]dense.Neuron)}
	for i := 0; i < numHiddenNeurons; i++ {
		neuronID := "hidden" + strconv.Itoa(i)
		newNeuron := dense.Neuron{
			ActivationType: "relu", // Using ReLU for hidden layer
			Connections:    make(map[string]dense.Connection),
			Bias:           rand.Float64(),
		}

		// Connect input neurons to hidden neurons
		for j := 0; j < numInputs; j++ {
			inputID := "input" + strconv.Itoa(j)
			newNeuron.Connections[inputID] = dense.Connection{Weight: rand.Float64()}
		}

		hiddenLayer.Neurons[neuronID] = newNeuron
	}

	// Add the hidden layer to the network
	network.Layers.Hidden = []dense.Layer{hiddenLayer}

	// Initialize output neurons with connections to hidden neurons
	network.Layers.Output.Neurons = make(map[string]dense.Neuron)
	for i := 0; i < numOutputs; i++ {
		neuronID := "output" + strconv.Itoa(i)
		activationType := "sigmoid" // Default activation function
		if i < len(outputActivationTypes) {
			activationType = outputActivationTypes[i]
		}

		outputNeuron := dense.Neuron{
			ActivationType: activationType, // Custom or default activation for output
			Connections:    make(map[string]dense.Connection),
			Bias:           rand.Float64(),
		}

		// Connect hidden neurons to output neurons
		for j := 0; j < numHiddenNeurons; j++ {
			hiddenID := "hidden" + strconv.Itoa(j)
			outputNeuron.Connections[hiddenID] = dense.Connection{Weight: rand.Float64()}
		}

		network.Layers.Output.Neurons[neuronID] = outputNeuron
	}

	return network
}
