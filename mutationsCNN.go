package dense

import (
	"fmt"
	"math/rand"
)

func MutateCNNWeights(config *NetworkConfig, learningRate float64, mutationRate int) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" {
			for i := range layer.Filters {
				if rand.Intn(100) < mutationRate {
					for j := range layer.Filters[i].Weights {
						for k := range layer.Filters[i].Weights[j] {
							layer.Filters[i].Weights[j][k] += rand.NormFloat64() * learningRate
						}
					}
				}
			}
		}
	}
}

func MutateCNNBiases(config *NetworkConfig, mutationRate int, learningRate float64) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" {
			for i := range layer.Filters {
				if rand.Intn(100) < mutationRate {
					layer.Filters[i].Bias += rand.NormFloat64() * learningRate
				}
			}
		}
	}
}

func RandomizeCNNWeights(config *NetworkConfig, mutationRate int) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" {
			for i := range layer.Filters {
				if rand.Intn(100) < mutationRate {
					layer.Filters[i].Weights = Random2DSlice(len(layer.Filters[i].Weights), len(layer.Filters[i].Weights[0]))
				}
			}
		}
	}
}

func Random2DSlice(rows, cols int) [][]float64 {
	slice := make([][]float64, rows)
	for i := range slice {
		slice[i] = RandomSlice(cols)
	}
	return slice
}

func InvertCNNWeights(config *NetworkConfig, mutationRate int) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" {
			for i := range layer.Filters {
				if rand.Intn(100) < mutationRate {
					for j := range layer.Filters[i].Weights {
						for k := range layer.Filters[i].Weights[j] {
							layer.Filters[i].Weights[j][k] = -layer.Filters[i].Weights[j][k]
						}
					}
				}
			}
		}
	}
}

// AddCNNLayerAtRandomPosition adds a new CNN layer with a random filter size at a random position.
func AddCNNLayerAtRandomPosition(config *NetworkConfig, mutationRate int) {
	if rand.Intn(100) < mutationRate {
		// Define possible filter sizes (e.g., 3x3, 5x5, 7x7, etc.)
		//filterSizes := []int{3, 5, 7, 11, 13}

		// Randomly select a filter size from the list
		//randomFilterSize := filterSizes[rand.Intn(len(filterSizes))]
		randomFilterSize := 3 + rand.Intn(1022)

		newLayer := Layer{
			LayerType: "conv",
			Filters: []Filter{
				{
					Weights: Random2DSlice(randomFilterSize, randomFilterSize),
					Bias:    rand.Float64(),
				},
			},
			Stride:  1,
			Padding: 1,
		}

		// Insert at a random position in the hidden layers
		pos := rand.Intn(len(config.Layers.Hidden) + 1)
		config.Layers.Hidden = append(config.Layers.Hidden[:pos], append([]Layer{newLayer}, config.Layers.Hidden[pos:]...)...)

		//fmt.Printf("Added CNN layer with %dx%d filters at position %d.\n", randomFilterSize, randomFilterSize, pos)
	}
}

// MutateCNNFilterSize mutates the size of convolution filters.
func MutateCNNFilterSize(config *NetworkConfig, mutationRate int) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" {
			for i := range layer.Filters {
				if rand.Intn(100) < mutationRate {
					newSize := rand.Intn(3) + 3 // Random size between 3 and 5
					layer.Filters[i].Weights = Random2DSlice(newSize, newSize)
				}
			}
		}
	}
}

// MutateCNNStrideAndPadding mutates the stride and padding values of CNN layers.
func MutateCNNStrideAndPadding(config *NetworkConfig, mutationRate int) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" && rand.Intn(100) < mutationRate {
			layer.Stride = rand.Intn(3) + 1 // Random stride between 1 and 3
			layer.Padding = rand.Intn(2)    // Random padding between 0 and 1
		}
	}
}

// DuplicateCNNLayer duplicates a random convolutional layer.
func DuplicateCNNLayer(config *NetworkConfig, mutationRate int) {
	if rand.Intn(100) < mutationRate && len(config.Layers.Hidden) > 0 {
		pos := rand.Intn(len(config.Layers.Hidden))
		layerToDuplicate := config.Layers.Hidden[pos]
		newLayer := layerToDuplicate // Shallow copy of the layer

		// Insert the duplicated layer after the original one
		config.Layers.Hidden = append(config.Layers.Hidden[:pos+1], append([]Layer{newLayer}, config.Layers.Hidden[pos+1:]...)...)
	}
}

// AddMultipleCNNLayers adds a random number of new convolutional layers with random filter size, stride, and padding.
func AddMultipleCNNLayers(config *NetworkConfig, mutationRate int, maxLayers int) {
	if rand.Intn(100) < mutationRate {
		// Randomize the number of layers to add, between 1 and maxLayers
		numLayers := rand.Intn(maxLayers) + 1

		// Possible filter sizes to choose from
		//filterSizes := []int{3, 5, 7, 11, 13}

		for i := 0; i < numLayers; i++ {
			randomFilterSize := 3 + rand.Intn(1022)
			// Randomize filter size, stride, and padding for each layer
			//randomFilterSize := filterSizes[rand.Intn(len(filterSizes))]
			randomStride := rand.Intn(3) + 1 // Random stride between 1 and 3
			randomPadding := rand.Intn(2)    // Random padding between 0 and 1

			newLayer := Layer{
				LayerType: "conv",
				Filters: []Filter{
					{
						Weights: Random2DSlice(randomFilterSize, randomFilterSize),
						Bias:    rand.Float64(),
					},
				},
				Stride:  randomStride,
				Padding: randomPadding,
			}

			// Insert the new layer at a random position
			pos := rand.Intn(len(config.Layers.Hidden) + 1)
			config.Layers.Hidden = append(config.Layers.Hidden[:pos], append([]Layer{newLayer}, config.Layers.Hidden[pos:]...)...)

			//fmt.Printf("Added CNN layer with %dx%d filters, stride %d, padding %d at position %d.\n", randomFilterSize, randomFilterSize, randomStride, randomPadding, pos)
		}
	}
}

// AppendCNNLayer appends a new CNN layer with specified parameters to the Hidden layers in NetworkConfig.
// Parameters:
// - config: Pointer to the NetworkConfig to be modified.
// - filterSize: Size of the convolutional filters (e.g., 3 for 3x3).
// - numFilters: Number of filters in the convolutional layer.
// - stride: Stride value for the convolution.
// - padding: Padding value to maintain spatial dimensions.
func AppendCNNLayer(config *NetworkConfig, filterSize int, numFilters int, stride int, padding int) error {
	// Validate filter size
	if filterSize <= 0 {
		return fmt.Errorf("invalid filter size: %d. Must be positive", filterSize)
	}

	// Validate number of filters
	if numFilters <= 0 {
		return fmt.Errorf("invalid number of filters: %d. Must be positive", numFilters)
	}

	// Validate stride
	if stride <= 0 {
		return fmt.Errorf("invalid stride: %d. Must be positive", stride)
	}

	// Validate padding
	if padding < 0 {
		return fmt.Errorf("invalid padding: %d. Must be non-negative", padding)
	}

	// Create and initialize filters
	filters := make([]Filter, numFilters)
	for i := 0; i < numFilters; i++ {
		filters[i] = Filter{
			Weights: Random2DSlice(filterSize, filterSize),
			Bias:    rand.Float64(),
		}
	}

	// Create the new convolutional layer
	newLayer := Layer{
		LayerType: "conv",
		Filters:   filters,
		Stride:    stride,
		Padding:   padding,
	}

	// Append the new layer to the Hidden layers
	config.Layers.Hidden = append(config.Layers.Hidden, newLayer)

	// Log the addition
	fmt.Printf("Appended CNN layer with %dx%d filters, %d filters, stride %d, padding %d.\n",
		filterSize, filterSize, numFilters, stride, padding)

	return nil
}

// AppendCNNAndDenseLayer appends a CNN layer followed by a Dense layer to the NetworkConfig.
// Parameters:
// - config: Pointer to the NetworkConfig to be modified.
// - filterSize: Size of the convolutional filters (e.g., 3 for 3x3).
// - numFilters: Number of filters in the convolutional layer.
// - stride: Stride value for the convolution.
// - padding: Padding value to maintain spatial dimensions.
// - denseNeurons: Number of neurons in the Dense layer.
func AppendCNNAndDenseLayer(config *NetworkConfig, filterSize int, numFilters int, stride int, padding int, denseNeurons int) error {
	// Append CNN Layer
	err := AppendCNNLayer(config, filterSize, numFilters, stride, padding)
	if err != nil {
		return fmt.Errorf("failed to append CNN layer: %v", err)
	}

	// Determine the naming convention for CNN outputs
	// Assuming each filter produces one feature map, and each feature map is flattened
	// We'll name them as "conv_output_filterX_elementY"

	// For simplicity, let's assume each feature map is flattened into a single value (this is a simplification)
	// In practice, you might need to handle multiple elements per feature map

	// Create Dense Layer
	denseLayer := Layer{
		LayerType: "dense",
		Neurons:   make(map[string]Neuron),
	}

	for i := 0; i < denseNeurons; i++ {
		neuronID := fmt.Sprintf("dense_neuron_%d", i)
		connections := make(map[string]Connection)

		// Connect to each filter's bias (as a simplification)
		// In a real scenario, you might need to flatten feature maps and connect accordingly
		for j := 0; j < numFilters; j++ {
			connectionID := fmt.Sprintf("conv_filter_%d_bias", j)
			connections[connectionID] = Connection{Weight: rand.Float64() - 0.5} // Initialize weights around 0
		}

		denseLayer.Neurons[neuronID] = Neuron{
			ActivationType: "relu", // You can parameterize this if needed
			Connections:    connections,
			Bias:           rand.Float64() - 0.5, // Initialize biases around 0
		}
	}

	// Append the Dense Layer
	config.Layers.Hidden = append(config.Layers.Hidden, denseLayer)

	// Log the addition
	/*fmt.Printf("Appended CNN layer with %dx%d filters, %d filters, stride %d, padding %d.\n",
		filterSize, filterSize, numFilters, stride, padding)
	fmt.Printf("Appended Dense layer with %d neurons.\n", denseNeurons)*/

	return nil
}
