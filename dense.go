package dense

import (
	"math"
	"math/rand"
	"strconv"
)

// Connection represents a connection between two neurons with a weight.
type Connection struct {
	Weight float64 `json:"weight"`
}

// Neuron represents a neuron with an activation function, connections, and a bias.
type Neuron struct {
	ActivationType string                `json:"activationType"`
	Connections    map[string]Connection `json:"connections"`
	Bias           float64               `json:"bias"`
}

// Layer represents a layer of neurons in the network.
type Layer struct {
	Neurons map[string]Neuron `json:"neurons"`
}

// NetworkConfig represents the structure of the neural network, containing input, hidden, and output layers.
type NetworkConfig struct {
	Layers struct {
		Input  Layer   `json:"input"`
		Hidden []Layer `json:"hidden"`
		Output Layer   `json:"output"`
	} `json:"layers"`
}

// Activate function calculates the activation value based on the activation type.
func activate(activationType string, input float64) float64 {
	switch activationType {
	case "relu":
		return math.Max(0, input)
	case "sigmoid":
		return 1 / (1 + math.Exp(-input))
	case "tanh":
		return math.Tanh(input)
	case "softmax":
		return math.Exp(input) // Should normalize later in the layer processing
	case "leaky_relu":
		if input > 0 {
			return input
		}
		return 0.01 * input
	case "swish":
		return input * (1 / (1 + math.Exp(-input))) // Beta set to 1 for simplicity
	case "elu":
		alpha := 1.0 // Alpha can be adjusted based on specific needs
		if input >= 0 {
			return input
		}
		return alpha * (math.Exp(input) - 1)
	case "selu":
		lambda := 1.0507    // Scale factor
		alphaSELU := 1.6733 // Alpha for SELU
		if input >= 0 {
			return lambda * input
		}
		return lambda * (alphaSELU * (math.Exp(input) - 1))
	case "softplus":
		return math.Log(1 + math.Exp(input))
	default:
		return input // Linear activation (no change)
	}
}

// Feedforward processes the input values through the network and returns the output values.
func Feedforward(config *NetworkConfig, inputValues map[string]float64) map[string]float64 {
	neurons := make(map[string]float64)

	// Load input values into the neurons
	for inputID := range config.Layers.Input.Neurons {
		neurons[inputID] = inputValues[inputID]
	}

	// Process hidden layers
	for _, layer := range config.Layers.Hidden {
		for nodeID, node := range layer.Neurons {
			sum := 0.0
			for inputID, connection := range node.Connections {
				sum += neurons[inputID] * connection.Weight
			}
			sum += node.Bias
			neurons[nodeID] = activate(node.ActivationType, sum)
		}
	}

	// Process output layer
	outputs := make(map[string]float64)
	for nodeID, node := range config.Layers.Output.Neurons {
		sum := 0.0
		for inputID, connection := range node.Connections {
			sum += neurons[inputID] * connection.Weight
		}
		sum += node.Bias
		outputs[nodeID] = activate(node.ActivationType, sum)
	}

	return outputs
}

// RandomWeight generates a random weight for connections.
func RandomWeight() float64 {
	return rand.NormFloat64()
}

// CreateRandomNetworkConfig dynamically generates a network with specified input and output sizes and allows dynamic configuration of output neurons.
func CreateRandomNetworkConfig(numInputs, numOutputs int, outputActivationTypes []string) *NetworkConfig {
	config := &NetworkConfig{}

	// Define input neurons
	config.Layers.Input.Neurons = make(map[string]Neuron)
	for i := 0; i < numInputs; i++ {
		neuronID := "input" + strconv.Itoa(i)
		config.Layers.Input.Neurons[neuronID] = Neuron{}
	}

	// Define hidden layers with random weights and biases
	config.Layers.Hidden = []Layer{
		{
			Neurons: map[string]Neuron{
				"hidden1": {
					ActivationType: "relu",
					Connections: func() map[string]Connection {
						connections := make(map[string]Connection)
						for i := 0; i < numInputs; i++ {
							connections["input"+strconv.Itoa(i)] = Connection{Weight: rand.Float64()}
						}
						return connections
					}(),
					Bias: rand.Float64(),
				},
			},
		},
	}

	// Define output neurons with customizable activation types, random weights, and biases
	config.Layers.Output.Neurons = make(map[string]Neuron)
	for i := 0; i < numOutputs; i++ {
		neuronID := "output" + strconv.Itoa(i)
		activationType := "sigmoid" // Default activation function
		if i < len(outputActivationTypes) {
			activationType = outputActivationTypes[i]
		}

		config.Layers.Output.Neurons[neuronID] = Neuron{
			ActivationType: activationType,
			Connections: map[string]Connection{
				"hidden1": {Weight: rand.Float64()},
			},
			Bias: rand.Float64(),
		}
	}

	return config
}

// DeepCopy creates a deep copy of the NetworkConfig, so each goroutine works on its own copy.
func DeepCopy(config *NetworkConfig) *NetworkConfig {
	// Initialize a new config with the correct struct type (including json tags)
	newConfig := &NetworkConfig{
		Layers: struct {
			Input  Layer   `json:"input"`
			Hidden []Layer `json:"hidden"`
			Output Layer   `json:"output"`
		}{
			Input: Layer{
				Neurons: make(map[string]Neuron),
			},
			Hidden: make([]Layer, len(config.Layers.Hidden)),
			Output: Layer{
				Neurons: make(map[string]Neuron),
			},
		},
	}

	// Copy input layer neurons
	for key, neuron := range config.Layers.Input.Neurons {
		newNeuron := Neuron{
			ActivationType: neuron.ActivationType,
			Bias:           neuron.Bias,
			Connections:    make(map[string]Connection),
		}
		for connKey, conn := range neuron.Connections {
			newNeuron.Connections[connKey] = conn
		}
		newConfig.Layers.Input.Neurons[key] = newNeuron
	}

	// Copy hidden layers and neurons
	for i, layer := range config.Layers.Hidden {
		newLayer := Layer{
			Neurons: make(map[string]Neuron),
		}
		for key, neuron := range layer.Neurons {
			newNeuron := Neuron{
				ActivationType: neuron.ActivationType,
				Bias:           neuron.Bias,
				Connections:    make(map[string]Connection),
			}
			for connKey, conn := range neuron.Connections {
				newNeuron.Connections[connKey] = conn
			}
			newLayer.Neurons[key] = newNeuron
		}
		newConfig.Layers.Hidden[i] = newLayer
	}

	// Copy output layer neurons
	for key, neuron := range config.Layers.Output.Neurons {
		newNeuron := Neuron{
			ActivationType: neuron.ActivationType,
			Bias:           neuron.Bias,
			Connections:    make(map[string]Connection),
		}
		for connKey, conn := range neuron.Connections {
			newNeuron.Connections[connKey] = conn
		}
		newConfig.Layers.Output.Neurons[key] = newNeuron
	}

	return newConfig
}
