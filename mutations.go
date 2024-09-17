package dense

import (
	"fmt"
	"math/rand"
	"time"
)

// MutationType defines the types of mutations available
type MutationType int
const (
    MutateWeight MutationType = iota
    AddNeuronMutation
    AddLayerFullConnectionMutation
    AddLayerSparseMutation
    AddLayerRandomPositionMutation
    MutateActivationFunction
    RemoveNeuronMutation
    RemoveLayerMutation
    DuplicateNeuronMutation
    MutateBiasMutation
    RandomizeWeightsMutation
    SplitNeuronMutation
    SwapLayerActivationsMutation
    ShuffleLayerConnectionsMutation
    ShuffleLayersMutation // New mutation type to shuffle layers
)
// Example usage in MutateNetwork
func MutateNetwork(config *NetworkConfig, learningRate float64, mutationRate int) {
    rand.Seed(time.Now().UnixNano())

    // Randomly select the mutation type to apply
    switch rand.Intn(34) { // Updated to include the new mutation types
    case int(MutateWeight):
        MutateWeights(config, learningRate, mutationRate)
    case int(AddNeuronMutation):
        AddNeuron(config, mutationRate)
    case int(AddLayerFullConnectionMutation):
        AddLayerFullConnections(config, mutationRate)
    case int(AddLayerSparseMutation):
        AddLayer(config, mutationRate)
    case int(AddLayerRandomPositionMutation):
        AddLayerRandomPosition(config, mutationRate)
    case int(MutateActivationFunction):
        MutateActivationFunctions(config, mutationRate)
    case int(RemoveNeuronMutation):
        RemoveNeuron(config, mutationRate)
    case int(RemoveLayerMutation):
        RemoveLayer(config, mutationRate)
    case int(DuplicateNeuronMutation):
        DuplicateNeuron(config, mutationRate)
    case int(MutateBiasMutation):
        MutateBiases(config, mutationRate, learningRate)
    case int(RandomizeWeightsMutation):
        RandomizeWeights(config, mutationRate)
    case int(SplitNeuronMutation):
        SplitNeuron(config, mutationRate)
    case int(SwapLayerActivationsMutation):
        SwapLayerActivations(config, mutationRate)
    case int(ShuffleLayerConnectionsMutation):
        ShuffleLayerConnections(config, mutationRate)
    case int(ShuffleLayersMutation):
        ShuffleLayers(config, mutationRate)
    case 15: // Add multiple random layers
        AddMultipleLayers(config, mutationRate)
    case 16: // Double the number of layers
        DoubleLayers(config, mutationRate)
    case 17: // Mirror layers from top to bottom
        MirrorLayersTopToBottom(config, mutationRate)
    case 18: // Mirror edges from side to side
        MirrorEdgesSideToSide(config, mutationRate)
    case 19: // Invert weights
        InvertWeights(config, mutationRate)
    case 20: // Invert biases
        InvertBiases(config, mutationRate)
    case 21: // Invert activation functions
        InvertActivationFunctions(config, mutationRate)
    case 22: // Invert connections
        InvertConnections(config, mutationRate)

    // LSTM mutations
    case 23: 
        MutateLSTMCells(config, mutationRate)
    case 24: 
        AddLSTMLayerAtRandomPosition(config, mutationRate)
    case 25:
        InvertLSTMWeights(config, mutationRate)
    case 26:
        RandomizeLSTMWeights(config, mutationRate)
    case 27:
        MutateLSTMBiases(config, mutationRate, learningRate) // Fixed argument mismatch
    case 28:
        MutateLSTMWeights(config, learningRate, mutationRate) // Fixed argument mismatch

    // CNN mutations
    case 29:
        MutateCNNWeights(config, learningRate, mutationRate) // Fixed argument mismatch
    case 30:
        MutateCNNBiases(config, mutationRate, learningRate)  // Fixed argument mismatch
    case 31:
        RandomizeCNNWeights(config, mutationRate) // Fixed argument mismatch
    case 32:
        InvertCNNWeights(config, mutationRate)

    case 33:
        AddCNNLayerAtRandomPosition(config, mutationRate)
    }

    // restoreInputAndOutputLayers(config, savedInputLayer, savedOutputLayer)
}



// InvertWeights inverts a percentage of the network's weights based on the mutation rate
func InvertWeights(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            for connID, conn := range neuron.Connections {
                if rand.Intn(100) < mutationRate {
                    conn.Weight = -conn.Weight // Invert the weight
                    neuron.Connections[connID] = conn
                }
            }
            layer.Neurons[neuronID] = neuron
        }
    }

    for neuronID, neuron := range config.Layers.Output.Neurons {
        for connID, conn := range neuron.Connections {
            if rand.Intn(100) < mutationRate {
                conn.Weight = -conn.Weight // Invert the weight
                neuron.Connections[connID] = conn
            }
        }
        config.Layers.Output.Neurons[neuronID] = neuron
    }

    //fmt.Printf("Inverted weights with mutation rate of %d%%.\n", mutationRate)
}

// InvertBiases inverts a percentage of the neuron biases based on the mutation rate
func InvertBiases(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            if rand.Intn(100) < mutationRate {
                neuron.Bias = -neuron.Bias // Invert the bias
                layer.Neurons[neuronID] = neuron
            }
        }
    }

    for neuronID, neuron := range config.Layers.Output.Neurons {
        if rand.Intn(100) < mutationRate {
            neuron.Bias = -neuron.Bias // Invert the bias
            config.Layers.Output.Neurons[neuronID] = neuron
        }
    }

    //fmt.Printf("Inverted biases with mutation rate of %d%%.\n", mutationRate)
}

// InvertActivationFunctions inverts the activation functions based on mutation rate
func InvertActivationFunctions(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    activationInversionMap := map[string]string{
        "relu":       "leaky_relu", // Replace with an "opposite" or alternate activation
        "sigmoid":    "tanh",
        "tanh":       "sigmoid",
        "leaky_relu": "relu",
    }

    // Randomly mutate activation functions for neurons in hidden layers
    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            if rand.Intn(100) < mutationRate {
                invertedActivation := activationInversionMap[neuron.ActivationType]
                neuron.ActivationType = invertedActivation
                layer.Neurons[neuronID] = neuron
            }
        }
    }

    // Randomly mutate activation functions for neurons in output layer
    for neuronID, neuron := range config.Layers.Output.Neurons {
        if rand.Intn(100) < mutationRate {
            invertedActivation := activationInversionMap[neuron.ActivationType]
            neuron.ActivationType = invertedActivation
            config.Layers.Output.Neurons[neuronID] = neuron
        }
    }

    //fmt.Printf("Inverted activation functions with mutation rate of %d%%.\n", mutationRate)
}

// InvertConnections inverts a percentage of connections between neurons based on mutation rate
func InvertConnections(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            invertedConnections := make(map[string]Connection)
            for connID, conn := range neuron.Connections {
                if rand.Intn(100) < mutationRate {
                    invertedConnections[connID] = Connection{Weight: -conn.Weight} // Invert the connection weight
                } else {
                    invertedConnections[connID] = conn
                }
            }
            neuron.Connections = invertedConnections
            layer.Neurons[neuronID] = neuron
        }
    }

    for neuronID, neuron := range config.Layers.Output.Neurons {
        invertedConnections := make(map[string]Connection)
        for connID, conn := range neuron.Connections {
            if rand.Intn(100) < mutationRate {
                invertedConnections[connID] = Connection{Weight: -conn.Weight} // Invert the connection weight
            } else {
                invertedConnections[connID] = conn
            }
        }
        neuron.Connections = invertedConnections
        config.Layers.Output.Neurons[neuronID] = neuron
    }

    //fmt.Printf("Inverted neuron connections with mutation rate of %d%%.\n", mutationRate)
}


// AddMultipleLayers adds a random number of layers to the network
func AddMultipleLayers(config *NetworkConfig, mutationRate int) {
    rand.Seed(time.Now().UnixNano())

    if rand.Intn(100) < mutationRate {
        numNewLayers := rand.Intn(5) + 1 // Add 1 to 5 layers randomly
        for i := 0; i < numNewLayers; i++ {
            newLayer := Layer{
                Neurons: make(map[string]Neuron),
            }

            // Add 1 to 3 neurons to each new layer
            numNewNeurons := rand.Intn(3) + 1
            for j := 0; j < numNewNeurons; j++ {
                neuronID := fmt.Sprintf("neuron%d", len(newLayer.Neurons)+1)
                newNeuron := Neuron{
                    ActivationType: randomActivationType(),
                    Connections:    make(map[string]Connection),
                    Bias:           rand.NormFloat64(),
                }

                // Connect the new neuron to the previous layer's neurons
                var previousLayerNeurons map[string]Neuron
                if len(config.Layers.Hidden) == 0 {
                    previousLayerNeurons = config.Layers.Input.Neurons
                } else {
                    previousLayerNeurons = config.Layers.Hidden[len(config.Layers.Hidden)-1].Neurons
                }
                for prevNeuronID := range previousLayerNeurons {
                    newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
                }

                // Add the new neuron to the layer
                newLayer.Neurons[neuronID] = newNeuron
            }

            // Append the new layer to the hidden layers
            config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
        }

        //fmt.Printf("Added %d new layers to the network.\n", numNewLayers)
    }
}

// DoubleLayers duplicates the current layers in the network
func DoubleLayers(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        currentLayers := len(config.Layers.Hidden)
        for i := 0; i < currentLayers; i++ {
            newLayer := Layer{
                Neurons: make(map[string]Neuron),
            }

            // Duplicate neurons
            for neuronID, neuron := range config.Layers.Hidden[i].Neurons {
                newNeuronID := fmt.Sprintf("%s_dup", neuronID)
                newLayer.Neurons[newNeuronID] = neuron
            }

            // Append the duplicated layer to the hidden layers
            config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
        }

        //fmt.Printf("Doubled the number of layers, now %d layers in total.\n", len(config.Layers.Hidden))
    }
}

// MirrorLayersTopToBottom mirrors the layers from top to bottom (reverse the order)
func MirrorLayersTopToBottom(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        mirroredLayers := make([]Layer, len(config.Layers.Hidden))
        for i := range config.Layers.Hidden {
            mirroredLayers[len(config.Layers.Hidden)-1-i] = config.Layers.Hidden[i]
        }

        // Append mirrored layers
        config.Layers.Hidden = append(config.Layers.Hidden, mirroredLayers...)
        //fmt.Printf("Mirrored the layers from top to bottom.\n")
    }
}

// MirrorEdgesSideToSide mirrors the connections in each layer from side to side (reverse the connections)
func MirrorEdgesSideToSide(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        for _, layer := range config.Layers.Hidden {
            for neuronID, neuron := range layer.Neurons {
                mirroredConnections := make(map[string]Connection)
                for connID, conn := range neuron.Connections {
                    mirroredConnID := fmt.Sprintf("%s_mirrored", connID)
                    mirroredConnections[mirroredConnID] = conn
                }

                // Append mirrored connections to the neuron
                for mirroredConnID, mirroredConn := range mirroredConnections {
                    neuron.Connections[mirroredConnID] = mirroredConn
                }
                layer.Neurons[neuronID] = neuron
            }
        }

        //fmt.Printf("Mirrored the edges in each layer from side to side.\n")
    }
}

// ShuffleLayers shuffles the order of hidden layers based on the mutation rate.
func ShuffleLayers(config *NetworkConfig, mutationRate int) {
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return
    }

    // Decide how many layers to shuffle based on the mutation rate.
    numLayersToShuffle := int(float64(len(config.Layers.Hidden)) * float64(mutationRate) / 100.0)
    
    if numLayersToShuffle == 0 {
        return
    }

    // Generate shuffled indices and reorder the hidden layers.
    indices := rand.Perm(len(config.Layers.Hidden))
    shuffledLayers := make([]Layer, len(config.Layers.Hidden))

    for i := 0; i < numLayersToShuffle; i++ {
        shuffledLayers[i] = config.Layers.Hidden[indices[i]]
    }

    // Update the hidden layers with the shuffled order.
    config.Layers.Hidden = shuffledLayers

    //fmt.Printf("Shuffled %d layers based on mutation rate of %d%%\n", numLayersToShuffle, mutationRate)
}




// MutateWeights randomly mutates the network's weights with a given mutation rate
func OLDMutateWeights(config *NetworkConfig, learningRate float64, mutationRate int) {
    rand.Seed(time.Now().UnixNano())

    // Ensure mutationRate is within bounds
    if mutationRate < 0 {
        mutationRate = 0
    } else if mutationRate > 100 {
        mutationRate = 100
    }

    // Step 1: Count total number of weights and biases
    totalWeights := 0
    for _, layer := range config.Layers.Hidden {
        for _, neuron := range layer.Neurons {
            totalWeights += len(neuron.Connections) // Add the number of connections (weights)
            totalWeights++                         // Count the bias as part of the weights
        }
    }
    for _, neuron := range config.Layers.Output.Neurons {
        totalWeights += len(neuron.Connections) // Add the number of connections (weights)
        totalWeights++                         // Count the bias as part of the weights
    }

    // Step 2: Calculate how many weights to mutate based on mutation rate
    weightsToMutate := int(float64(totalWeights) * (float64(mutationRate) / 100.0))

    // Step 3: Randomly choose which weights to mutate
    mutatedCount := 0
    for _, layer := range config.Layers.Hidden {
        for _, neuron := range layer.Neurons {
            for connID := range neuron.Connections {
                if mutatedCount >= weightsToMutate {
                    break
                }
                // Randomly decide if we mutate this connection (weight)
                if rand.Float64() < float64(weightsToMutate-mutatedCount)/float64(totalWeights-mutatedCount) {
                    neuron.Connections[connID] = Connection{
                        Weight: neuron.Connections[connID].Weight + rand.NormFloat64()*learningRate,
                    }
                    mutatedCount++
                }
            }
            if mutatedCount >= weightsToMutate {
                break
            }
            // Randomly decide if we mutate this neuron's bias
            if mutatedCount < weightsToMutate && rand.Float64() < float64(weightsToMutate-mutatedCount)/float64(totalWeights-mutatedCount) {
                neuron.Bias += rand.NormFloat64() * learningRate
                mutatedCount++
            }
        }
    }

    // Mutate output layer weights and biases
    for _, neuron := range config.Layers.Output.Neurons {
        for connID := range neuron.Connections {
            if rand.Float64() < float64(mutationRate)/100.0 {
                neuron.Connections[connID] = Connection{
                    Weight: neuron.Connections[connID].Weight + rand.NormFloat64()*learningRate,
                }
            }
        }
        // Mutate output neuron bias
        if rand.Float64() < float64(mutationRate)/100.0 {
            neuron.Bias += rand.NormFloat64() * learningRate
        }
    }
}

func MutateWeights(config *NetworkConfig, learningRate float64, mutationRate int) {
    rand.Seed(time.Now().UnixNano())

    if mutationRate <= 0 {
        return
    }

    // Mutate only dense (FFNN) or convolutional (CNN) layers
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" { // FFNN layer
            for neuronID, neuron := range layer.Neurons {
                for connID := range neuron.Connections {
                    if rand.Intn(100) < mutationRate {
                        neuron.Connections[connID] = Connection{
                            Weight: neuron.Connections[connID].Weight + rand.NormFloat64()*learningRate,
                        }
                    }
                }
                neuron.Bias += rand.NormFloat64() * learningRate
                layer.Neurons[neuronID] = neuron
            }
        } else if layer.LayerType == "conv" { // CNN layer
            for i := range layer.Filters {
                filter := &layer.Filters[i]
                for x := range filter.Weights {
                    for y := range filter.Weights[x] {
                        if rand.Intn(100) < mutationRate {
                            filter.Weights[x][y] += rand.NormFloat64() * learningRate
                        }
                    }
                }
                filter.Bias += rand.NormFloat64() * learningRate
            }
        }
    }

    // Also mutate the output layer (which should be FFNN)
    if config.Layers.Output.LayerType == "dense" {
        for neuronID, neuron := range config.Layers.Output.Neurons {
            for connID := range neuron.Connections {
                if rand.Intn(100) < mutationRate {
                    neuron.Connections[connID] = Connection{
                        Weight: neuron.Connections[connID].Weight + rand.NormFloat64()*learningRate,
                    }
                }
            }
            neuron.Bias += rand.NormFloat64() * learningRate
            config.Layers.Output.Neurons[neuronID] = neuron
        }
    }
}


// AddNeuron adds a new neuron to a random hidden layer based on the mutation rate
func OLDAddNeuron(config *NetworkConfig, mutationRate int) {
    // Ensure mutationRate is within bounds
    if mutationRate < 0 {
        mutationRate = 0
    } else if mutationRate > 100 {
        mutationRate = 100
    }

    // Check if there are any hidden layers to add a neuron to
    if len(config.Layers.Hidden) == 0 {
       //  fmt.Println("No hidden layers found. Adding a new layer first.")
        AddLayer(config, mutationRate) // Add a new layer if there are none
        return
    }

    // Randomly decide if we should add a neuron based on the mutation rate
    if rand.Intn(100) < mutationRate {
        // Randomly pick an existing hidden layer to add the neuron to
        layerIdx := rand.Intn(len(config.Layers.Hidden))
        layer := &config.Layers.Hidden[layerIdx]

        // Add the new neuron
        neuronID := fmt.Sprintf("neuron%d", len(layer.Neurons)+1)
        newNeuron := Neuron{
            ActivationType: randomActivationType(),
            Connections:    make(map[string]Connection),
            Bias:           rand.NormFloat64(),
        }

        // Connect the new neuron to the previous layer
        var previousLayerNeurons map[string]Neuron
        if layerIdx == 0 {
            previousLayerNeurons = config.Layers.Input.Neurons
        } else {
            previousLayerNeurons = config.Layers.Hidden[layerIdx-1].Neurons
        }
        for prevNeuronID := range previousLayerNeurons {
            newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
        }

        // Connect the new neuron to the next layer (if it exists)
        if layerIdx < len(config.Layers.Hidden)-1 {
            nextLayer := &config.Layers.Hidden[layerIdx+1]
            for nextNeuronID := range nextLayer.Neurons {
                nextLayer.Neurons[nextNeuronID].Connections[neuronID] = Connection{Weight: rand.NormFloat64()}
            }
        } else {
            // Connect the new neuron to the output layer
            for outputNeuronID := range config.Layers.Output.Neurons {
                config.Layers.Output.Neurons[outputNeuronID].Connections[neuronID] = Connection{Weight: rand.NormFloat64()}
            }
        }

        // Add the new neuron to the selected layer
        layer.Neurons[neuronID] = newNeuron

       //  fmt.Printf("Added a new neuron to hidden layer %d\n", layerIdx+1)
    }
}

func AddNeuron(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    // Only add neurons to dense (FFNN) layers
    for i, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" && rand.Intn(100) < mutationRate {
            neuronID := fmt.Sprintf("neuron%d", len(layer.Neurons)+1)
            newNeuron := Neuron{
                ActivationType: randomActivationType(),
                Connections:    make(map[string]Connection),
                Bias:           rand.NormFloat64(),
            }

            // Connect the new neuron to the previous layer
            var previousLayerNeurons map[string]Neuron
            if i == 0 {
                previousLayerNeurons = config.Layers.Input.Neurons
            } else {
                previousLayerNeurons = config.Layers.Hidden[i-1].Neurons
            }

            for prevNeuronID := range previousLayerNeurons {
                newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
            }

            // Add the neuron to the current layer
            layer.Neurons[neuronID] = newNeuron
            config.Layers.Hidden[i] = layer
        }
    }
}



// AddLayer adds a new hidden layer with random neurons to the network
func AddLayerFullConnections(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        newLayer := Layer{
            Neurons: make(map[string]Neuron),
        }

        // Add 1 to 3 neurons to this new layer
        numNewNeurons := rand.Intn(3) + 1
        for i := 0; i < numNewNeurons; i++ {
            neuronID := fmt.Sprintf("neuron%d", len(newLayer.Neurons)+1)
            newNeuron := Neuron{
                ActivationType: randomActivationType(),
                Connections:    make(map[string]Connection),
                Bias:           rand.NormFloat64(),
            }

            // Connect the new neuron to the previous layer's neurons
            var previousLayerNeurons map[string]Neuron
            if len(config.Layers.Hidden) == 0 {
                previousLayerNeurons = config.Layers.Input.Neurons
            } else {
                previousLayerNeurons = config.Layers.Hidden[len(config.Layers.Hidden)-1].Neurons
            }
            for prevNeuronID := range previousLayerNeurons {
                newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
            }

            // Add the new neuron to the layer
            newLayer.Neurons[neuronID] = newNeuron
           //  fmt.Printf("Added neuron %s to new layer with connections to previous layer\n", neuronID)
        }

        // Connect the new layer's neurons to the output layer (or next hidden layer if one exists)
        if len(config.Layers.Hidden) == 0 {
           //  fmt.Println("Connecting new layer to the output layer directly")
        }
        //for outputNeuronID, outputNeuron := range config.Layers.Output.Neurons {
        for _, outputNeuron := range config.Layers.Output.Neurons {
            for newNeuronID := range newLayer.Neurons {
                outputNeuron.Connections[newNeuronID] = Connection{Weight: rand.NormFloat64()}
               //  fmt.Printf("Connecting new neuron %s to output neuron %s\n", newNeuronID, outputNeuronID)
            }
        }

        // Append the new layer to the hidden layers
        config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
       //  fmt.Printf("Added a new hidden layer with %d neurons\n", numNewNeurons)
    }
}


// AddLayer adds a new hidden layer with random sparse connections
func OLDAddLayer(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        newLayer := Layer{
            Neurons: make(map[string]Neuron),
        }

        // Add 1 to 3 neurons to this new layer
        numNewNeurons := rand.Intn(3) + 1
        for i := 0; i < numNewNeurons; i++ {
            neuronID := fmt.Sprintf("neuron%d", len(newLayer.Neurons)+1)
            newNeuron := Neuron{
                ActivationType: randomActivationType(),
                Connections:    make(map[string]Connection),
                Bias:           rand.NormFloat64(),
            }

            // Connect the new neuron to a random subset of the previous layer's neurons
            var previousLayerNeurons map[string]Neuron
            if len(config.Layers.Hidden) == 0 {
                previousLayerNeurons = config.Layers.Input.Neurons
            } else {
                previousLayerNeurons = config.Layers.Hidden[len(config.Layers.Hidden)-1].Neurons
            }

            // Generate a random connection ratio between 0 and 1 for sparse connections
            connectionRatio := rand.Float64()

            // Create sparse connections based on random connectionRatio
            for prevNeuronID := range previousLayerNeurons {
                if rand.Float64() < connectionRatio {  // Random connection
                    newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
                }
            }

            // Add the new neuron to the layer
            newLayer.Neurons[neuronID] = newNeuron
           //  fmt.Printf("Added neuron %s to new layer with random sparse connections (ratio: %.2f)\n", neuronID, connectionRatio)
        }

        // Connect the new layer's neurons to the output layer (or next hidden layer if one exists)
        if len(config.Layers.Hidden) == 0 {
           //  fmt.Println("Connecting new layer to the output layer directly")
        }
        //for outputNeuronID, outputNeuron := range config.Layers.Output.Neurons {
        for _, outputNeuron := range config.Layers.Output.Neurons {
            for newNeuronID := range newLayer.Neurons {
                outputNeuron.Connections[newNeuronID] = Connection{Weight: rand.NormFloat64()}
               //  fmt.Printf("Connecting new neuron %s to output neuron %s\n", newNeuronID, outputNeuronID)
            }
        }

        // Append the new layer to the hidden layers
        config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
       //  fmt.Printf("Added a new hidden layer with %d neurons\n", numNewNeurons)
    }
}

func AddLayer(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        newLayer := Layer{
            Neurons: make(map[string]Neuron),
        }

        // Only add FFNN or CNN layers
        if rand.Intn(2) == 0 {
            newLayer.LayerType = "dense" // FFNN
            for i := 0; i < rand.Intn(3)+1; i++ {
                neuronID := fmt.Sprintf("neuron%d", len(newLayer.Neurons)+1)
                newLayer.Neurons[neuronID] = Neuron{
                    ActivationType: randomActivationType(),
                    Bias:           rand.Float64(),
                    Connections:    make(map[string]Connection),
                }
            }
        } else {
            newLayer.LayerType = "conv" // CNN
            newLayer.Filters = append(newLayer.Filters, Filter{
                Weights: Random2DSlice(3, 3),
                Bias:    rand.Float64(),
            })
        }

        // Append to hidden layers
        config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
    }
}



// AddLayer adds a new hidden layer with random sparse connections at a random position
func AddLayerRandomPosition(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        newLayer := Layer{
            Neurons: make(map[string]Neuron),
        }

        // Add 1 to 3 neurons to this new layer
        numNewNeurons := rand.Intn(3) + 1
        for i := 0; i < numNewNeurons; i++ {
            neuronID := fmt.Sprintf("neuron%d", len(newLayer.Neurons)+1)
            newNeuron := Neuron{
                ActivationType: randomActivationType(),
                Connections:    make(map[string]Connection),
                Bias:           rand.NormFloat64(),
            }

            // Connect the new neuron to a random subset of previous layer's neurons
            var previousLayerNeurons map[string]Neuron
            if len(config.Layers.Hidden) == 0 {
                previousLayerNeurons = config.Layers.Input.Neurons
            } else {
                previousLayerNeurons = config.Layers.Hidden[len(config.Layers.Hidden)-1].Neurons
            }

            // Generate a random connection ratio between 0 and 1 for sparse connections
            connectionRatio := rand.Float64()

            // Create sparse connections based on random connectionRatio
            for prevNeuronID := range previousLayerNeurons {
                if rand.Float64() < connectionRatio {
                    newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
                }
            }

            // Add the new neuron to the layer
            newLayer.Neurons[neuronID] = newNeuron
           //  fmt.Printf("Added neuron %s to new layer with random sparse connections (ratio: %.2f)\n", neuronID, connectionRatio)
        }

        // Randomly choose a position to insert the new layer
        insertPosition := rand.Intn(len(config.Layers.Hidden) + 1)

        // Insert the new layer at the randomly chosen position
        config.Layers.Hidden = append(config.Layers.Hidden[:insertPosition], append([]Layer{newLayer}, config.Layers.Hidden[insertPosition:]...)...)
       //  fmt.Printf("Inserted a new hidden layer with %d neurons at position %d\n", numNewNeurons, insertPosition+1)
    }
}

// MutateActivationFunctions randomizes the activation functions for all neurons based on the mutation rate
func OLDMutateActivationFunctions(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    // Randomly mutate activation functions for neurons in hidden layers
    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            // Randomly decide if we mutate this neuron's activation function
            if rand.Intn(100) < mutationRate {
                newActivation := randomActivationType()
                neuron.ActivationType = newActivation
                layer.Neurons[neuronID] = neuron // Apply the mutation
               //  fmt.Printf("Mutated activation function of neuron %s to %s\n", neuronID, newActivation)
            }
        }
    }

    // Randomly mutate activation functions for neurons in output layer
    for neuronID, neuron := range config.Layers.Output.Neurons {
        if rand.Intn(100) < mutationRate {
            newActivation := randomActivationType()
            neuron.ActivationType = newActivation
            config.Layers.Output.Neurons[neuronID] = neuron // Apply the mutation
           //  fmt.Printf("Mutated activation function of output neuron %s to %s\n", neuronID, newActivation)
        }
    }
}

func MutateActivationFunctions(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    // Mutate only dense (FFNN) layers
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" {
            for neuronID, neuron := range layer.Neurons {
                if rand.Intn(100) < mutationRate {
                    neuron.ActivationType = randomActivationType()
                    layer.Neurons[neuronID] = neuron
                }
            }
        }
    }

    // Also mutate the output layer (which is likely FFNN)
    if config.Layers.Output.LayerType == "dense" {
        for neuronID, neuron := range config.Layers.Output.Neurons {
            if rand.Intn(100) < mutationRate {
                neuron.ActivationType = randomActivationType()
                config.Layers.Output.Neurons[neuronID] = neuron
            }
        }
    }
}





// Helper function to choose a random activation function
func randomActivationType() string {
    activationTypes := []string{"relu", "sigmoid", "tanh", "leaky_relu"}
    return activationTypes[rand.Intn(len(activationTypes))]
}



func RemoveNeuron(config *NetworkConfig, mutationRate int) {
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return
    }
    for layerIdx := range config.Layers.Hidden {
        if rand.Intn(100) < mutationRate && len(config.Layers.Hidden[layerIdx].Neurons) > 1 {
            // Randomly select a neuron to remove
            for neuronID := range config.Layers.Hidden[layerIdx].Neurons {
                delete(config.Layers.Hidden[layerIdx].Neurons, neuronID)
               //  fmt.Printf("Removed neuron %s from hidden layer %d\n", neuronID, layerIdx+1)
                break
            }
            break
        }
    }
}

func RemoveLayer(config *NetworkConfig, mutationRate int) {
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return
    }
    if rand.Intn(100) < mutationRate {
        layerIdx := rand.Intn(len(config.Layers.Hidden))
        config.Layers.Hidden = append(config.Layers.Hidden[:layerIdx], config.Layers.Hidden[layerIdx+1:]...)
       //  fmt.Printf("Removed hidden layer at position %d\n", layerIdx+1)
    }
}


func DuplicateNeuron(config *NetworkConfig, mutationRate int) {
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return
    }
    for layerIdx := range config.Layers.Hidden {
        if rand.Intn(100) < mutationRate {
            // Randomly select a neuron to duplicate
            for neuronID, neuron := range config.Layers.Hidden[layerIdx].Neurons {
                newNeuronID := fmt.Sprintf("%s_dup", neuronID)
                config.Layers.Hidden[layerIdx].Neurons[newNeuronID] = neuron
               //  fmt.Printf("Duplicated neuron %s as %s in hidden layer %d\n", neuronID, newNeuronID, layerIdx+1)
                break
            }
            break
        }
    }
}


func MutateBiases(config *NetworkConfig, mutationRate int, learningRate float64) {
    if mutationRate <= 0 {
        return
    }

    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            if rand.Intn(100) < mutationRate {
                neuron.Bias += rand.NormFloat64() * learningRate
                layer.Neurons[neuronID] = neuron
               //  fmt.Printf("Mutated bias of neuron %s to %.4f\n", neuronID, neuron.Bias)
            }
        }
    }

    for neuronID, neuron := range config.Layers.Output.Neurons {
        if rand.Intn(100) < mutationRate {
            neuron.Bias += rand.NormFloat64() * learningRate
            config.Layers.Output.Neurons[neuronID] = neuron
           //  fmt.Printf("Mutated bias of output neuron %s to %.4f\n", neuronID, neuron.Bias)
        }
    }
}

func RandomizeWeights(config *NetworkConfig, mutationRate int) {
    if mutationRate <= 0 {
        return
    }

    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            for connID := range neuron.Connections {
                if rand.Intn(100) < mutationRate {
                    neuron.Connections[connID] = Connection{Weight: rand.NormFloat64()}
                   //  fmt.Printf("Randomized weight of connection %s for neuron %s\n", connID, neuronID)
                }
            }
            layer.Neurons[neuronID] = neuron
        }
    }

    for neuronID, neuron := range config.Layers.Output.Neurons {
        for connID := range neuron.Connections {
            if rand.Intn(100) < mutationRate {
                neuron.Connections[connID] = Connection{Weight: rand.NormFloat64()}
               //  fmt.Printf("Randomized weight of connection %s for output neuron %s\n", connID, neuronID)
            }
        }
        config.Layers.Output.Neurons[neuronID] = neuron
    }
}


func SplitNeuron(config *NetworkConfig, mutationRate int) {
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return
    }
    for layerIdx := range config.Layers.Hidden {
        if rand.Intn(100) < mutationRate {
            for neuronID, neuron := range config.Layers.Hidden[layerIdx].Neurons {
                // Create two new neurons with a split of the connections
                newNeuron1 := neuron
                newNeuron2 := neuron
                halfConnections := len(neuron.Connections) / 2

                for connID := range neuron.Connections {
                    if halfConnections > 0 {
                        delete(newNeuron2.Connections, connID)
                        halfConnections--
                    } else {
                        delete(newNeuron1.Connections, connID)
                    }
                }

                config.Layers.Hidden[layerIdx].Neurons[neuronID+"_split1"] = newNeuron1
                config.Layers.Hidden[layerIdx].Neurons[neuronID+"_split2"] = newNeuron2

               //  fmt.Printf("Split neuron %s into %s_split1 and %s_split2\n", neuronID, neuronID, neuronID)
                delete(config.Layers.Hidden[layerIdx].Neurons, neuronID)
                break
            }
            break
        }
    }
}





func SwapLayerActivations(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate && len(config.Layers.Hidden) > 1 {
        // Randomly select two layers to swap activations
        idx1 := rand.Intn(len(config.Layers.Hidden))
        idx2 := rand.Intn(len(config.Layers.Hidden))
        
        if idx1 != idx2 {
            layer1 := config.Layers.Hidden[idx1]
            layer2 := config.Layers.Hidden[idx2]
            
            // Ensure both layers are of type "dense" and have neurons
            if layer1.LayerType == "dense" && layer2.LayerType == "dense" && layer1.Neurons != nil && layer2.Neurons != nil {
                for neuronID, neuron1 := range layer1.Neurons {
                    neuron2, ok := layer2.Neurons[neuronID]
                    if !ok {
                        continue // Ensure neuron exists in both layers
                    }
                    if neuron1.Connections == nil || neuron2.Connections == nil {
                        // Skip if connections are nil
                        continue
                    }
                    neuron1.ActivationType, neuron2.ActivationType = neuron2.ActivationType, neuron1.ActivationType
                    layer1.Neurons[neuronID], layer2.Neurons[neuronID] = neuron1, neuron2
                }
            }
        }
    }
}




func ShuffleLayerConnections(config *NetworkConfig, mutationRate int) {
    //for layerIdx, layer := range config.Layers.Hidden {
    for _, layer := range config.Layers.Hidden {
        if rand.Intn(100) < mutationRate {
            neuronIDs := make([]string, 0, len(layer.Neurons))
            for neuronID := range layer.Neurons {
                neuronIDs = append(neuronIDs, neuronID)
            }
            rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })
            
            // Shuffle the connections by copying and reassigning modified neurons
            for i, neuronID := range neuronIDs {
                neuron := layer.Neurons[neuronID] // Get the neuron struct from the map
                shuffledNeuron := layer.Neurons[neuronIDs[(i+1)%len(neuronIDs)]]
                neuron.Connections = shuffledNeuron.Connections // Assign shuffled connections
                layer.Neurons[neuronID] = neuron // Put the modified neuron back into the map
            }
            
           //  fmt.Printf("Shuffled connections in layer %d\n", layerIdx+1)
        }
    }
}



func SaveInputAndOutputLayers(config *NetworkConfig) (inputLayer, outputLayer Layer) {
    inputLayer = config.Layers.Input   // Save input layer
    outputLayer = config.Layers.Output // Save output layer
    return inputLayer, outputLayer
}

func RestoreInputAndOutputLayers(config *NetworkConfig, inputLayer, outputLayer Layer) {
    config.Layers.Input = inputLayer   // Restore input layer
    config.Layers.Output = outputLayer // Restore output layer
}

func CheckForLayerChanges(original, mutated Layer, layerType string) {
    // Check if the number of neurons is different
    if len(original.Neurons) != len(mutated.Neurons) {
        fmt.Printf("Warning: %s layer was altered during mutation! (Neuron count changed)\n", layerType)
        return
    }

    // Iterate over the neurons in the original layer and compare with the mutated layer
    for neuronID, originalNeuron := range original.Neurons {
        mutatedNeuron, exists := mutated.Neurons[neuronID]
        if !exists {
            fmt.Printf("Warning: %s layer was altered during mutation! (Neuron %s removed)\n", layerType, neuronID)
            return
        }

        // Check if the activation function changed
        if originalNeuron.ActivationType != mutatedNeuron.ActivationType {
            fmt.Printf("Warning: %s layer was altered during mutation! (Activation function of neuron %s changed)\n", layerType, neuronID)
            return
        }

        // Compare connections
        if len(originalNeuron.Connections) != len(mutatedNeuron.Connections) {
            fmt.Printf("Warning: %s layer was altered during mutation! (Connection count for neuron %s changed)\n", layerType, neuronID)
            return
        }

        for connID, originalConn := range originalNeuron.Connections {
            mutatedConn, connExists := mutatedNeuron.Connections[connID]
            if !connExists {
                fmt.Printf("Warning: %s layer was altered during mutation! (Connection %s for neuron %s removed)\n", layerType, connID, neuronID)
                return
            }

            // Optionally, check for weight changes
            if originalConn.Weight != mutatedConn.Weight {
                fmt.Printf("Warning: %s layer was altered during mutation! (Connection weight for neuron %s -> %s changed)\n", layerType, neuronID, connID)
                return
            }
        }
    }

    fmt.Printf("%s layer was not altered during mutation.\n", layerType)
}

