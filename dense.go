package dense

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"
	"sync"
)

type Connection struct {
	Weight float64 `json:"weight"`
}

type Neuron struct {
	ActivationType string                `json:"activationType"`
	Connections    map[string]Connection `json:"connections"`
	Bias           float64               `json:"bias"`
}

type Layer struct {
	Neurons map[string]Neuron `json:"neurons"`
}

type NetworkConfig struct {
    Layers struct {
        Input  Layer   `json:"input"`
        Hidden []Layer `json:"hidden"`
        Output Layer   `json:"output"`
    } `json:"layers"`

    // Add the mutex for thread-safe access
    mutex sync.RWMutex 
}


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

func Feedforward(config *NetworkConfig, inputValues map[string]float64) map[string]float64 {
    config.mutex.RLock() // Lock for reading
    defer config.mutex.RUnlock() // Unlock after reading

    neurons := make(map[string]float64)

    // Load inputs into the neuron map
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



func RandomizeModelOnlyLayer() string {
	rand.Seed(time.Now().UnixNano())
	activationTypes := []string{"relu", "sigmoid", "tanh", "softmax", "leaky_relu", "swish", "elu", "selu", "softplus"}
	activationType := activationTypes[rand.Intn(len(activationTypes))]

	weight1 := rand.NormFloat64()
	bias := rand.NormFloat64()

	model := map[string]interface{}{
		"layers": map[string]interface{}{
			"hidden": []map[string]interface{}{
				{
					"neurons": map[string]interface{}{
						"4": map[string]interface{}{
							"activationType": activationType,
							"connections": map[string]interface{}{
								"1": map[string]interface{}{
									"weight": weight1,
								},
							},
							"bias": bias,
						},
					},
				},
			},
		},
	}

	modelJSON, _ := json.Marshal(model)
	return string(modelJSON)
}

func RandomWeight() float64 {
	return rand.NormFloat64()
}

func RandomizeNetworkStaticTesting() string {
	model := map[string]interface{}{
		"layers": map[string]interface{}{
			"input": map[string]interface{}{
				"neurons": map[string]interface{}{
					"1": map[string]interface{}{},
					"2": map[string]interface{}{},
					"3": map[string]interface{}{},
				},
			},
			"hidden": []map[string]interface{}{
				{
					"neurons": map[string]interface{}{
						"4": map[string]interface{}{
							"activationType": "relu",
							"connections": map[string]interface{}{
								"1": map[string]interface{}{
									"weight": RandomWeight(),
								},
							},
							"bias": rand.Float64(),
						},
					},
				},
			},
			"output": map[string]interface{}{
				"neurons": map[string]interface{}{
					"5": map[string]interface{}{
						"activationType": "sigmoid",
						"connections": map[string]interface{}{
							"4": map[string]interface{}{
								"weight": RandomWeight(),
							},
						},
						"bias": rand.Float64(),
					},
					"6": map[string]interface{}{
						"activationType": "sigmoid",
						"connections": map[string]interface{}{
							"4": map[string]interface{}{
								"weight": RandomWeight(),
							},
						},
						"bias": rand.Float64(),
					},
					"7": map[string]interface{}{
						"activationType": "sigmoid",
						"connections": map[string]interface{}{
							"4": map[string]interface{}{
								"weight": RandomWeight(),
							},
						},
						"bias": rand.Float64(),
					},
				},
			},
		},
	}

	modelJSON, _ := json.MarshalIndent(model, "", "  ")
	return string(modelJSON)
}

// The following are the new additions based on the errors encountered

type TestData struct {
	Inputs  map[string]float64
	Outputs map[string]float64
}

func CreateTestNetworkConfig() *NetworkConfig {
	config := &NetworkConfig{}
	config.Layers.Input.Neurons = map[string]Neuron{
		"input1": {},
		"input2": {},
		"input3": {},
	}

	config.Layers.Hidden = []Layer{
		{
			Neurons: map[string]Neuron{
				"hidden1": {
					ActivationType: "relu",
					Connections: map[string]Connection{
						"input1": {Weight: 0.5},
						"input2": {Weight: 0.6},
						"input3": {Weight: 0.7},
					},
					Bias: 0.1,
				},
			},
		},
	}

	config.Layers.Output.Neurons = map[string]Neuron{
		"output1": {
			ActivationType: "sigmoid",
			Connections: map[string]Connection{
				"hidden1": {Weight: 0.8},
			},
			Bias: 0.2,
		},
		"output2": {
			ActivationType: "sigmoid",
			Connections: map[string]Connection{
				"hidden1": {Weight: 0.9},
			},
			Bias: 0.3,
		},
		"output3": {
			ActivationType: "sigmoid",
			Connections: map[string]Connection{
				"hidden1": {Weight: 1.0},
			},
			Bias: 0.4,
		},
	}

	return config
}

func TestNeuralNetwork(config *NetworkConfig, testData []TestData) {
	for i, data := range testData {
		outputs := Feedforward(config, data.Inputs)
		fmt.Printf("Test Case %d:\n", i+1)
		fmt.Printf("Inputs: %v\n", data.Inputs)
		fmt.Printf("Expected Outputs: %v\n", data.Outputs)
		fmt.Printf("Actual Outputs: %v\n\n", outputs)
	}
}


// Create a random neural network configuration for testing
func CreateRandomNetworkConfig() *NetworkConfig {
	config := &NetworkConfig{}
	config.Layers.Input.Neurons = map[string]Neuron{
		"input1": {},
		"input2": {},
		"input3": {},
	}

	config.Layers.Hidden = []Layer{
		{
			Neurons: map[string]Neuron{
				"hidden1": {
					ActivationType: "relu",
					Connections: map[string]Connection{
						"input1": {Weight: rand.Float64()},
						"input2": {Weight: rand.Float64()},
						"input3": {Weight: rand.Float64()},
					},
					Bias: rand.Float64(),
				},
			},
		},
	}

	config.Layers.Output.Neurons = map[string]Neuron{
		"output1": {
			ActivationType: "sigmoid",
			Connections: map[string]Connection{
				"hidden1": {Weight: rand.Float64()},
			},
			Bias: rand.Float64(),
		},
	}

	return config
}
