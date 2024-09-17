package dense

import (
	"fmt"
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

// Filter represents a convolutional filter (kernel).
type Filter struct {
	Weights [][]float64 `json:"weights"`
	Bias    float64     `json:"bias"`
}

// LSTMCell represents a cell in an LSTM layer.
type LSTMCell struct {
	InputWeights  []float64 `json:"inputWeights"`
	ForgetWeights []float64 `json:"forgetWeights"`
	OutputWeights []float64 `json:"outputWeights"`
	CellWeights   []float64 `json:"cellWeights"`
	Bias          float64   `json:"bias"`
}

// Layer represents a layer in the network.
type Layer struct {
	LayerType string            `json:"layerType"`
	Neurons   map[string]Neuron `json:"neurons,omitempty"` // For dense layers
	// For convolutional layers
	Filters []Filter `json:"filters,omitempty"`
	Stride  int      `json:"stride,omitempty"`
	Padding int      `json:"padding,omitempty"`
	// For LSTM layers
	LSTMCells []LSTMCell `json:"lstmCells,omitempty"`
}

// ModelMetadata holds metadata for the model.
type ModelMetadata struct {
	ModelID             string  `json:"modelID"`
	ProjectName         string  `json:"projectName"`
	LastTrainingAccuracy float64 `json:"lastTrainingAccuracy"`
	LastTestAccuracy     float64 `json:"lastTestAccuracy"`
}

// NetworkConfig represents the structure of the neural network, containing input, hidden, and output layers, and model metadata.
type NetworkConfig struct {
	Metadata ModelMetadata `json:"metadata"`
	Layers   struct {
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
func Feedforward(config *NetworkConfig, inputValues map[string]interface{}) map[string]float64 {
	var data interface{}

	// Load input values into the data variable
	if config.Layers.Input.LayerType == "dense" {
		// Input is dense
		// Convert inputValues to map[string]float64
		inputData := make(map[string]float64)
		for k, v := range inputValues {
			if val, ok := v.(float64); ok {
				inputData[k] = val
			} else {
				// Handle error
				return nil
			}
		}
		data = inputData
	} else if config.Layers.Input.LayerType == "conv" {
		// Input is convolutional
		if imageData, ok := inputValues["image"].([][]float64); ok {
			data = imageData
		} else {
			// Handle error
			return nil
		}
	} else if config.Layers.Input.LayerType == "lstm" {
		// Input is sequence
		if sequenceData, ok := inputValues["sequence"].([][]float64); ok {
			data = sequenceData
		} else {
			// Handle error
			return nil
		}
	}

	// Process hidden layers
	for _, layer := range config.Layers.Hidden {
		switch layer.LayerType {
		case "dense":
			data = processDenseLayer(layer, data)
		case "conv":
			data = processConvLayer(layer, data)
		case "lstm":
			data = processLSTMLayer(layer, data)
		default:
			// Handle error or default case
		}
	}

	// Process output layer
	outputLayer := config.Layers.Output
	switch outputLayer.LayerType {
	case "dense":
		data = processDenseLayer(outputLayer, data)
	case "conv":
		data = processConvLayer(outputLayer, data)
	case "lstm":
		data = processLSTMLayer(outputLayer, data)
	default:
		// Handle error or default case
	}

	// Return output values
	if outputData, ok := data.(map[string]float64); ok {
		return outputData
	} else {
		// Handle error or convert data to desired format
	}

	return nil // or appropriate return
}

func processDenseLayer(layer Layer, inputData interface{}) interface{} {
	// inputData is map[string]float64 or output from previous layer
	var inputValues map[string]float64
	if m, ok := inputData.(map[string]float64); ok {
		inputValues = m
	} else if m, ok := inputData.(map[string]interface{}); ok {
		// Convert map[string]interface{} to map[string]float64
		inputValues = make(map[string]float64)
		for k, v := range m {
			if val, ok := v.(float64); ok {
				inputValues[k] = val
			} else {
				// Handle error
				return nil
			}
		}
	} else {
		// Handle error
		return nil
	}

	neurons := make(map[string]float64)

	for nodeID, node := range layer.Neurons {
		sum := 0.0
		for inputID, connection := range node.Connections {
			sum += inputValues[inputID] * connection.Weight
		}
		sum += node.Bias
		neurons[nodeID] = activate(node.ActivationType, sum)
	}

	return neurons
}

func processConvLayer(layer Layer, inputData interface{}) interface{} {
	// inputData is expected to be [][]float64 (2D image) or [][][]float64 (multiple feature maps)
	inputImages, ok := inputData.([][][]float64)
	if !ok {
		// Try to convert single image to array of images
		if singleImage, ok := inputData.([][]float64); ok {
			inputImages = [][][]float64{singleImage}
		} else {
			// Handle error
			return nil
		}
	}

	outputFeatureMaps := [][][]float64{}

	for _, filter := range layer.Filters {
		featureMapsForFilter := [][][]float64{}
		for _, inputImage := range inputImages {
			featureMap := convolve(inputImage, filter.Weights, layer.Stride, layer.Padding)
			// Apply activation function to each element in featureMap
			for i := range featureMap {
				for j := range featureMap[i] {
					featureMap[i][j] = activate("relu", featureMap[i][j]+filter.Bias) // Assuming ReLU activation
				}
			}
			featureMapsForFilter = append(featureMapsForFilter, featureMap)
		}
		// For simplicity, just append them
		outputFeatureMaps = append(outputFeatureMaps, featureMapsForFilter...)
	}

	// Flatten outputFeatureMaps into map[string]float64
	flattenedOutput := make(map[string]float64)
	idx := 0
	for _, featureMap := range outputFeatureMaps {
		for i := range featureMap {
			for j := range featureMap[i] {
				key := fmt.Sprintf("conv_output%d", idx)
				flattenedOutput[key] = featureMap[i][j]
				idx++
			}
		}
	}

	return flattenedOutput
}

func convolve(input [][]float64, kernel [][]float64, stride int, padding int) [][]float64 {
	// Pad the input if padding > 0
	paddedInput := pad2D(input, padding)

	inputHeight := len(paddedInput)
	inputWidth := len(paddedInput[0])

	kernelHeight := len(kernel)
	kernelWidth := len(kernel[0])

	// Calculate output dimensions
	outputHeight := (inputHeight - kernelHeight) / stride + 1
	outputWidth := (inputWidth - kernelWidth) / stride + 1

	output := make([][]float64, outputHeight)
	for i := 0; i < outputHeight; i++ {
		output[i] = make([]float64, outputWidth)
		for j := 0; j < outputWidth; j++ {
			sum := 0.0
			for ki := 0; ki < kernelHeight; ki++ {
				for kj := 0; kj < kernelWidth; kj++ {
					sum += paddedInput[i*stride+ki][j*stride+kj] * kernel[ki][kj]
				}
			}
			output[i][j] = sum
		}
	}

	return output
}

func pad2D(input [][]float64, padding int) [][]float64 {
	if padding == 0 {
		return input
	}

	inputHeight := len(input)
	inputWidth := len(input[0])

	paddedHeight := inputHeight + 2*padding
	paddedWidth := inputWidth + 2*padding

	paddedInput := make([][]float64, paddedHeight)
	for i := range paddedInput {
		paddedInput[i] = make([]float64, paddedWidth)
	}

	for i := 0; i < inputHeight; i++ {
		for j := 0; j < inputWidth; j++ {
			paddedInput[i+padding][j+padding] = input[i][j]
		}
	}

	return paddedInput
}

func processLSTMLayer(layer Layer, inputData interface{}) interface{} {
	// inputData is expected to be [][]float64 (sequence) or map[string]float64 (single time step)
	var sequence [][]float64

	switch v := inputData.(type) {
	case [][]float64:
		sequence = v
	case map[string]float64:
		// Convert map to []float64
		inputSlice := make([]float64, len(v))
		i := 0
		for _, val := range v {
			inputSlice[i] = val
			i++
		}
		sequence = [][]float64{inputSlice}
	default:
		// Handle error
		return nil
	}

	// Initialize hidden state and cell state
	var hiddenState []float64
	var cellState []float64

	// Assuming all LSTM cells have the same dimensions
	numCells := len(layer.LSTMCells)
	hiddenState = make([]float64, numCells)
	cellState = make([]float64, numCells)

	for _, timeStepInput := range sequence {
		// For each LSTM cell, compute the new hidden state and cell state
		newHiddenState := make([]float64, numCells)
		newCellState := make([]float64, numCells)

		for i, cell := range layer.LSTMCells {
			// Compute input gate, forget gate, output gate, and cell candidate
			// Assuming weights and inputs are compatible
			inputGate := sigmoid(dotProduct(cell.InputWeights, timeStepInput) + cell.Bias)
			forgetGate := sigmoid(dotProduct(cell.ForgetWeights, timeStepInput) + cell.Bias)
			outputGate := sigmoid(dotProduct(cell.OutputWeights, timeStepInput) + cell.Bias)
			cellCandidate := tanh(dotProduct(cell.CellWeights, timeStepInput) + cell.Bias)

			newCellState[i] = forgetGate*cellState[i] + inputGate*cellCandidate
			newHiddenState[i] = outputGate * tanh(newCellState[i])
		}

		hiddenState = newHiddenState
		cellState = newCellState
	}

	// Return the final hidden state as a map[string]float64
	output := make(map[string]float64)
	for i, value := range hiddenState {
		output["lstm"+strconv.Itoa(i)] = value
	}

	return output
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func dotProduct(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		// Handle error
		return 0
	}
	sum := 0.0
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// CreateRandomNetworkConfig dynamically generates a network with specified input and output sizes and allows dynamic configuration of output neurons.
func CreateRandomNetworkConfig(numInputs, numOutputs int, outputActivationTypes []string, modelID, projectName string) *NetworkConfig {
	config := &NetworkConfig{
		Metadata: ModelMetadata{
			ModelID:             modelID,
			ProjectName:         projectName,
			LastTrainingAccuracy: 0.0,
			LastTestAccuracy:     0.0,
		},
	}

	// Define input layer
	config.Layers.Input = Layer{
		LayerType: "dense", // Or "conv" or "lstm" depending on the network
		Neurons:   make(map[string]Neuron),
	}
	for i := 0; i < numInputs; i++ {
		neuronID := "input" + strconv.Itoa(i)
		config.Layers.Input.Neurons[neuronID] = Neuron{}
	}

	// Define hidden layers
	config.Layers.Hidden = []Layer{
		{
			LayerType: "dense",
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
		// Add convolutional layer
		{
			LayerType: "conv",
			Filters: []Filter{
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
		// Add LSTM layer
		{
			LayerType: "lstm",
			LSTMCells: []LSTMCell{
				{
					InputWeights:  RandomSlice(numInputs),
					ForgetWeights: RandomSlice(numInputs),
					OutputWeights: RandomSlice(numInputs),
					CellWeights:   RandomSlice(numInputs),
					Bias:          rand.Float64(),
				},
			},
		},
	}

	// Define output neurons with customizable activation types, random weights, and biases
	config.Layers.Output = Layer{
		LayerType: "dense",
		Neurons:   make(map[string]Neuron),
	}
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

func RandomSlice(length int) []float64 {
	slice := make([]float64, length)
	for i := range slice {
		slice[i] = rand.Float64()
	}
	return slice
}

// DeepCopy creates a deep copy of the NetworkConfig, so each goroutine works on its own copy.
func DeepCopy(config *NetworkConfig) *NetworkConfig {
	newConfig := &NetworkConfig{
		Metadata: config.Metadata, // Copy metadata
		Layers: struct {
			Input  Layer   `json:"input"`
			Hidden []Layer `json:"hidden"`
			Output Layer   `json:"output"`
		}{
			Input: Layer{
				LayerType: config.Layers.Input.LayerType,
			},
			Hidden: make([]Layer, len(config.Layers.Hidden)),
			Output: Layer{
				LayerType: config.Layers.Output.LayerType,
			},
		},
	}

	// Deep copy input layer
	newConfig.Layers.Input = deepCopyLayer(config.Layers.Input)

	// Deep copy hidden layers
	for i, layer := range config.Layers.Hidden {
		newConfig.Layers.Hidden[i] = deepCopyLayer(layer)
	}

	// Deep copy output layer
	newConfig.Layers.Output = deepCopyLayer(config.Layers.Output)

	return newConfig
}

func deepCopyLayer(layer Layer) Layer {
    newLayer := Layer{
        LayerType: layer.LayerType,
    }
    
    switch layer.LayerType {
    case "dense":
        // Ensure the neurons and connections are deep copied
        newLayer.Neurons = make(map[string]Neuron)
        for key, neuron := range layer.Neurons {
            newNeuron := Neuron{
                ActivationType: neuron.ActivationType,
                Bias:           neuron.Bias,
                Connections:    make(map[string]Connection),
            }
            for connKey, conn := range neuron.Connections {
                newNeuron.Connections[connKey] = Connection{Weight: conn.Weight}
            }
            newLayer.Neurons[key] = newNeuron
        }
    case "conv":
        // Copy filters for convolutional layers
        newLayer.Filters = make([]Filter, len(layer.Filters))
        for i, filter := range layer.Filters {
            newWeights := make([][]float64, len(filter.Weights))
            for j := range filter.Weights {
                newWeights[j] = make([]float64, len(filter.Weights[j]))
                copy(newWeights[j], filter.Weights[j])
            }
            newLayer.Filters[i] = Filter{
                Weights: newWeights,
                Bias:    filter.Bias,
            }
        }
        newLayer.Stride = layer.Stride
        newLayer.Padding = layer.Padding
    case "lstm":
        // Copy LSTM cells for LSTM layers
        newLayer.LSTMCells = make([]LSTMCell, len(layer.LSTMCells))
        for i, cell := range layer.LSTMCells {
            newCell := LSTMCell{
                Bias:          cell.Bias,
                InputWeights:  make([]float64, len(cell.InputWeights)),
                ForgetWeights: make([]float64, len(cell.ForgetWeights)),
                OutputWeights: make([]float64, len(cell.OutputWeights)),
                CellWeights:   make([]float64, len(cell.CellWeights)),
            }
            copy(newCell.InputWeights, cell.InputWeights)
            copy(newCell.ForgetWeights, cell.ForgetWeights)
            copy(newCell.OutputWeights, cell.OutputWeights)
            copy(newCell.CellWeights, cell.CellWeights)
            newLayer.LSTMCells[i] = newCell
        }
    }

    return newLayer
}

