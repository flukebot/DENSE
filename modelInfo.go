package dense

// CountLayers returns the total number of layers in the model, including input, hidden, and output layers.
func CountLayers(config *NetworkConfig) int {
    // Input layer is considered as 1 layer
    layerCount := 1
    
    // Add the number of hidden layers
    layerCount += len(config.Layers.Hidden)
    
    // Output layer is also considered as 1 layer
    layerCount += 1

    return layerCount
}
