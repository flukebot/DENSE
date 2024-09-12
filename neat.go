package dense

import (
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
) *NetworkConfig {
	// Initialize the metadata
	network := &NetworkConfig{
		Metadata: ModelMetadata{
			ModelID:             modelID,
			ProjectName:         projectName,
			LastTrainingAccuracy: 0.0,
			LastTestAccuracy:     0.0,
		},
	}

	// Initialize input neurons based on the passed number of input neurons
	network.Layers.Input.Neurons = make(map[string]Neuron)
	for i := 0; i < numInputs; i++ {
		neuronID := "input" + strconv.Itoa(i)
		network.Layers.Input.Neurons[neuronID] = Neuron{}
	}

	// Initialize hidden layer with 3 neurons
	numHiddenNeurons := 3
	hiddenLayer := Layer{Neurons: make(map[string]Neuron)}
	for i := 0; i < numHiddenNeurons; i++ {
		neuronID := "hidden" + strconv.Itoa(i)
		newNeuron := Neuron{
			ActivationType: "relu", // Using ReLU for hidden layer
			Connections:    make(map[string]Connection),
			Bias:           rand.Float64(),
		}

		// Connect input neurons to hidden neurons
		for j := 0; j < numInputs; j++ {
			inputID := "input" + strconv.Itoa(j)
			newNeuron.Connections[inputID] = Connection{Weight: rand.Float64()}
		}

		hiddenLayer.Neurons[neuronID] = newNeuron
	}

	// Add the hidden layer to the network
	network.Layers.Hidden = []Layer{hiddenLayer}

	// Initialize output neurons with connections to hidden neurons
	network.Layers.Output.Neurons = make(map[string]Neuron)
	for i := 0; i < numOutputs; i++ {
		neuronID := "output" + strconv.Itoa(i)
		activationType := "sigmoid" // Default activation function
		if i < len(outputActivationTypes) {
			activationType = outputActivationTypes[i]
		}

		outputNeuron := Neuron{
			ActivationType: activationType, // Custom or default activation for output
			Connections:    make(map[string]Connection),
			Bias:           rand.Float64(),
		}

		// Connect hidden neurons to output neurons
		for j := 0; j < numHiddenNeurons; j++ {
			hiddenID := "hidden" + strconv.Itoa(j)
			outputNeuron.Connections[hiddenID] = Connection{Weight: rand.Float64()}
		}

		network.Layers.Output.Neurons[neuronID] = outputNeuron
	}

	return network
}
