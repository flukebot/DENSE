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



// Maintain separate lists of mutation functions for FFNN, LSTM, and CNN
var LSTffnnMutations = map[string]MutationFunc{
	"FFNN_MutateWeights": MutateWeights,
    "FFNN_AddNeuron": AddNeuron,
    "FFNN_AddLayerFullConnections": AddLayerFullConnections,
    "FFNN_AddLayer": AddLayer,
    "FFNN_AddLayerRandomPosition": AddLayerRandomPosition,
    "FFNN_MutateActivationFunctions": MutateActivationFunctions,
    "FFNN_RemoveNeuron": RemoveNeuron,
    "FFNN_RemoveLayer": RemoveLayer,
    "FFNN_DuplicateNeuron": DuplicateNeuron,
    "FFNN_MutateBiases": MutateBiases,
    "FFNN_RandomizeWeights": RandomizeWeights,
    "FFNN_SplitNeuron": SplitNeuron,
    "FFNN_SwapLayerActivations": SwapLayerActivations,
    "FFNN_ShuffleLayerConnections": ShuffleLayerConnections,
    "FFNN_ShuffleLayers": ShuffleLayers,
    "FFNN_AddMultipleLayers": AddMultipleLayers,
    "FFNN_DoubleLayers": DoubleLayers,
    "FFNN_MirrorLayersTopToBottom": MirrorLayersTopToBottom,
    "FFNN_MirrorEdgesSideToSide": MirrorEdgesSideToSide,
    "FFNN_InvertWeights": InvertWeights,
    "FFNN_InvertBiases": InvertBiases,
    "FFNN_InvertActivationFunctions": InvertActivationFunctions,
    "FFNN_InvertConnections": InvertConnections,
}

var LSTlstmMutations = map[string]MutationFunc{
	"LSTM_MutateLSTMWeights": MutateLSTMWeights,
    "LSTM_MutateLSTMBiases": MutateLSTMBiases,
    "LSTM_RandomizeLSTMWeights": RandomizeLSTMWeights,
    "LSTM_InvertLSTMWeights": InvertLSTMWeights,
    "LSTM_AddLSTMLayerAtRandomPosition": AddLSTMLayerAtRandomPosition,
    "LSTM_MutateLSTMCells": MutateLSTMCells,
}

var LSTcnnMutations = map[string]MutationFunc{
	"CNN_MutateCNNWeights": MutateCNNWeights,
    "CNN_MutateCNNBiases": MutateCNNBiases,
    "CNN_RandomizeCNNWeights": RandomizeCNNWeights,
    "CNN_InvertCNNWeights": InvertCNNWeights,
    "CNN_AddCNNLayerAtRandomPosition": AddCNNLayerAtRandomPosition,
}

// Define a common function signature for all mutation functions
type MutationFunc func(args ...interface{}) error



func AddFFNNLayer(config *NetworkConfig) error {
    newLayer := Layer{
        LayerType: "dense",
        Neurons:   make(map[string]Neuron),
    }

    // Define the number of neurons to add in this layer
    numNeurons := rand.Intn(3) + 1 // Adding 1 to 3 neurons, adjust as necessary

    for i := 0; i < numNeurons; i++ {
        neuronID := fmt.Sprintf("neuron%d", len(newLayer.Neurons)+1)
        newNeuron := Neuron{
            ActivationType: randomActivationType(),
            Bias:           rand.Float64(),
            Connections:    make(map[string]Connection),
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

        newLayer.Neurons[neuronID] = newNeuron
    }

    config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
    return nil
}



func InvertWeights(args ...interface{}) error {
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for InvertWeights")
    }

    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if mutationRate <= 0 {
        return nil
    }

    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" {
            for neuronID, neuron := range layer.Neurons {
                for connID, conn := range neuron.Connections {
                    if rand.Intn(100) < mutationRate {
                        conn.Weight = -conn.Weight
                        neuron.Connections[connID] = conn
                    }
                }
                layer.Neurons[neuronID] = neuron
            }
        }
    }

    if config.Layers.Output.LayerType == "dense" {
        for neuronID, neuron := range config.Layers.Output.Neurons {
            for connID, conn := range neuron.Connections {
                if rand.Intn(100) < mutationRate {
                    conn.Weight = -conn.Weight
                    neuron.Connections[connID] = conn
                }
            }
            config.Layers.Output.Neurons[neuronID] = neuron
        }
    }

    return nil
}


func InvertBiases(args ...interface{}) error {
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for InvertBiases")
    }

    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if mutationRate <= 0 {
        return nil
    }

    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" {
            for neuronID, neuron := range layer.Neurons {
                if rand.Intn(100) < mutationRate {
                    neuron.Bias = -neuron.Bias
                    layer.Neurons[neuronID] = neuron
                }
            }
        }
    }

    if config.Layers.Output.LayerType == "dense" {
        for neuronID, neuron := range config.Layers.Output.Neurons {
            if rand.Intn(100) < mutationRate {
                neuron.Bias = -neuron.Bias
                config.Layers.Output.Neurons[neuronID] = neuron
            }
        }
    }

    return nil
}


// InvertActivationFunctions inverts the activation functions based on mutation rate
func InvertActivationFunctions(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for InvertActivationFunctions")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Check if mutationRate is valid
    if mutationRate <= 0 {
        return nil
    }

    // Define the activation inversion mapping
    activationInversionMap := map[string]string{
        "relu":       "leaky_relu",
        "sigmoid":    "tanh",
        "tanh":       "sigmoid",
        "leaky_relu": "relu",
    }

    // Randomly mutate activation functions for neurons in hidden layers
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" {
            for neuronID, neuron := range layer.Neurons {
                if rand.Intn(100) < mutationRate {
                    invertedActivation := activationInversionMap[neuron.ActivationType]
                    neuron.ActivationType = invertedActivation
                    layer.Neurons[neuronID] = neuron
                }
            }
        }
    }

    // Randomly mutate activation functions for neurons in output layer
    if config.Layers.Output.LayerType == "dense" {
        for neuronID, neuron := range config.Layers.Output.Neurons {
            if rand.Intn(100) < mutationRate {
                invertedActivation := activationInversionMap[neuron.ActivationType]
                neuron.ActivationType = invertedActivation
                config.Layers.Output.Neurons[neuronID] = neuron
            }
        }
    }

    return nil
}


// InvertConnections inverts a percentage of connections between neurons based on mutation rate
func InvertConnections(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for InvertConnections")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Check if mutationRate is valid
    if mutationRate <= 0 {
        return nil
    }

    // Invert connections for neurons in hidden layers
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" {
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
    }

    // Invert connections for neurons in the output layer
    if config.Layers.Output.LayerType == "dense" {
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
    }

    return nil
}



// AddMultipleLayers adds a random number of layers to the network
func AddMultipleLayers(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for AddMultipleLayers")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    rand.Seed(time.Now().UnixNano())

    // Check if we should apply the mutation based on the mutation rate
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

        // Optionally, you can log or return information about the added layers here.
        // fmt.Printf("Added %d new layers to the network.\n", numNewLayers)
    }

    return nil
}


// DoubleLayers duplicates the current layers in the network
func DoubleLayers(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for DoubleLayers")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

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

        // Optionally, you can log or return information about the doubled layers.
        // fmt.Printf("Doubled the number of layers, now %d layers in total.\n", len(config.Layers.Hidden))
    }

    return nil
}


// MirrorLayersTopToBottom mirrors the layers from top to bottom (reverse the order)
func MirrorLayersTopToBottom(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for MirrorLayersTopToBottom")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if rand.Intn(100) < mutationRate {
        mirroredLayers := make([]Layer, len(config.Layers.Hidden))
        for i := range config.Layers.Hidden {
            mirroredLayers[len(config.Layers.Hidden)-1-i] = config.Layers.Hidden[i]
        }

        // Append mirrored layers
        config.Layers.Hidden = append(config.Layers.Hidden, mirroredLayers...)
        // Optionally, you can log or return information about the mirrored layers.
        // fmt.Printf("Mirrored the layers from top to bottom.\n")
    }

    return nil
}


// MirrorEdgesSideToSide mirrors the connections in each layer from side to side (reverse the connections)
func MirrorEdgesSideToSide(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for MirrorEdgesSideToSide")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if rand.Intn(100) < mutationRate {
        for _, layer := range config.Layers.Hidden {
            if layer.LayerType == "dense" {
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
        }

        // Optionally, log the mirroring of connections.
        // fmt.Printf("Mirrored the edges in each layer from side to side.\n")
    }

    return nil
}


// ShuffleLayers shuffles the order of hidden layers based on the mutation rate.
func ShuffleLayers(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for ShuffleLayers")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return nil
    }

    // Decide how many layers to shuffle based on the mutation rate.
    numLayersToShuffle := int(float64(len(config.Layers.Hidden)) * float64(mutationRate) / 100.0)

    if numLayersToShuffle == 0 {
        return nil
    }

    // Generate shuffled indices and reorder the hidden layers.
    indices := rand.Perm(len(config.Layers.Hidden))
    shuffledLayers := make([]Layer, len(config.Layers.Hidden))

    for i := 0; i < numLayersToShuffle; i++ {
        shuffledLayers[i] = config.Layers.Hidden[indices[i]]
    }

    // Update the hidden layers with the shuffled order.
    config.Layers.Hidden = shuffledLayers

    // Optionally, log the shuffling of layers.
    // fmt.Printf("Shuffled %d layers based on mutation rate of %d%%\n", numLayersToShuffle, mutationRate)

    return nil
}


func MutateWeights(args ...interface{}) error {
    if len(args) < 3 {
        return fmt.Errorf("insufficient arguments for MutateWeights")
    }

    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    learningRate, ok := args[1].(float64)
    if !ok {
        return fmt.Errorf("invalid type for learningRate")
    }

    mutationRate, ok := args[2].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if mutationRate <= 0 {
        return nil
    }

    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" {
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
        }
    }

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

    return nil
}


func AddNeuron(args ...interface{}) error {
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for AddNeuron")
    }

    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if mutationRate <= 0 {
        return nil
    }

    for i, layer := range config.Layers.Hidden {
        if layer.LayerType == "dense" && rand.Intn(100) < mutationRate {
            neuronID := fmt.Sprintf("neuron%d", len(layer.Neurons)+1)
            newNeuron := Neuron{
                ActivationType: randomActivationType(),
                Connections:    make(map[string]Connection),
                Bias:           rand.NormFloat64(),
            }

            var previousLayerNeurons map[string]Neuron
            if i == 0 {
                previousLayerNeurons = config.Layers.Input.Neurons
            } else {
                previousLayerNeurons = config.Layers.Hidden[i-1].Neurons
            }

            for prevNeuronID := range previousLayerNeurons {
                newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
            }

            layer.Neurons[neuronID] = newNeuron
            config.Layers.Hidden[i] = layer
        }
    }

    return nil
}



// AddLayerFullConnections adds a new hidden layer with random neurons to the network
func AddLayerFullConnections(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for AddLayerFullConnections")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

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
        }

        // Connect the new layer's neurons to the output layer (or next hidden layer if one exists)
        for _, outputNeuron := range config.Layers.Output.Neurons {
            for newNeuronID := range newLayer.Neurons {
                outputNeuron.Connections[newNeuronID] = Connection{Weight: rand.NormFloat64()}
            }
        }

        // Append the new layer to the hidden layers
        config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
    }

    return nil
}



func AddLayer(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for AddLayer")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

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

    return nil
}



func AddLayerRandomPosition(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for AddLayerRandomPosition")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

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
        }

        // Randomly choose a position to insert the new layer
        insertPosition := rand.Intn(len(config.Layers.Hidden) + 1)

        // Insert the new layer at the randomly chosen position
        config.Layers.Hidden = append(config.Layers.Hidden[:insertPosition], append([]Layer{newLayer}, config.Layers.Hidden[insertPosition:]...)...)
    }

    return nil
}

func MutateActivationFunctions(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for MutateActivationFunctions")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if mutationRate <= 0 {
        return nil
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

    return nil
}






// Helper function to choose a random activation function
func randomActivationType() string {
    activationTypes := []string{"relu", "sigmoid", "tanh", "leaky_relu"}
    return activationTypes[rand.Intn(len(activationTypes))]
}



func RemoveNeuron(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for RemoveNeuron")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Check if there are any hidden layers and valid mutation rate
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return nil
    }

    // Iterate over the hidden layers to randomly remove a neuron
    for layerIdx := range config.Layers.Hidden {
        if rand.Intn(100) < mutationRate && len(config.Layers.Hidden[layerIdx].Neurons) > 1 {
            // Randomly select a neuron to remove
            for neuronID := range config.Layers.Hidden[layerIdx].Neurons {
                delete(config.Layers.Hidden[layerIdx].Neurons, neuronID)
                break
            }
            break
        }
    }

    return nil
}


func RemoveLayer(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for RemoveLayer")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Check if there are hidden layers and valid mutation rate
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return nil
    }

    // Remove a random layer based on mutation rate
    if rand.Intn(100) < mutationRate {
        layerIdx := rand.Intn(len(config.Layers.Hidden))
        config.Layers.Hidden = append(config.Layers.Hidden[:layerIdx], config.Layers.Hidden[layerIdx+1:]...)
    }

    return nil
}



func DuplicateNeuron(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for DuplicateNeuron")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Check if there are hidden layers and valid mutation rate
    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return nil
    }

    // Duplicate a neuron based on the mutation rate
    for layerIdx := range config.Layers.Hidden {
        if rand.Intn(100) < mutationRate {
            // Randomly select a neuron to duplicate
            for neuronID, neuron := range config.Layers.Hidden[layerIdx].Neurons {
                newNeuronID := fmt.Sprintf("%s_dup", neuronID)
                config.Layers.Hidden[layerIdx].Neurons[newNeuronID] = neuron
                break
            }
            break
        }
    }

    return nil
}



func MutateBiases(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 3 {
        return fmt.Errorf("insufficient arguments for MutateBiases")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Typecast the third argument to float64 for learningRate
    learningRate, ok := args[2].(float64)
    if !ok {
        return fmt.Errorf("invalid type for learningRate")
    }

    if mutationRate <= 0 {
        return nil
    }

    // Mutate biases in hidden layers
    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            if rand.Intn(100) < mutationRate {
                neuron.Bias += rand.NormFloat64() * learningRate
                layer.Neurons[neuronID] = neuron
            }
        }
    }

    // Mutate biases in output layer
    for neuronID, neuron := range config.Layers.Output.Neurons {
        if rand.Intn(100) < mutationRate {
            neuron.Bias += rand.NormFloat64() * learningRate
            config.Layers.Output.Neurons[neuronID] = neuron
        }
    }

    return nil
}


func RandomizeWeights(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for RandomizeWeights")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if mutationRate <= 0 {
        return nil
    }

    // Randomize weights in hidden layers
    for _, layer := range config.Layers.Hidden {
        for neuronID, neuron := range layer.Neurons {
            for connID := range neuron.Connections {
                if rand.Intn(100) < mutationRate {
                    neuron.Connections[connID] = Connection{Weight: rand.NormFloat64()}
                }
            }
            layer.Neurons[neuronID] = neuron
        }
    }

    // Randomize weights in output layer
    for neuronID, neuron := range config.Layers.Output.Neurons {
        for connID := range neuron.Connections {
            if rand.Intn(100) < mutationRate {
                neuron.Connections[connID] = Connection{Weight: rand.NormFloat64()}
            }
        }
        config.Layers.Output.Neurons[neuronID] = neuron
    }

    return nil
}



func SplitNeuron(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for SplitNeuron")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    if len(config.Layers.Hidden) == 0 || mutationRate <= 0 {
        return nil
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

                // Remove the original neuron
                delete(config.Layers.Hidden[layerIdx].Neurons, neuronID)
                break
            }
            break
        }
    }

    return nil
}


func SwapLayerActivations(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for SwapLayerActivations")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

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
                    // Swap the activation types
                    neuron1.ActivationType, neuron2.ActivationType = neuron2.ActivationType, neuron1.ActivationType
                    layer1.Neurons[neuronID], layer2.Neurons[neuronID] = neuron1, neuron2
                }
            }
        }
    }

    return nil
}





func ShuffleLayerConnections(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for ShuffleLayerConnections")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to int for mutationRate
    mutationRate, ok := args[1].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Loop through each hidden layer
    for _, layer := range config.Layers.Hidden {
        if rand.Intn(100) < mutationRate {
            neuronIDs := make([]string, 0, len(layer.Neurons))
            for neuronID := range layer.Neurons {
                neuronIDs = append(neuronIDs, neuronID)
            }
            // Shuffle the neuron IDs
            rand.Shuffle(len(neuronIDs), func(i, j int) { neuronIDs[i], neuronIDs[j] = neuronIDs[j], neuronIDs[i] })

            // Shuffle the connections by copying and reassigning modified neurons
            for i, neuronID := range neuronIDs {
                neuron := layer.Neurons[neuronID] // Get the neuron struct from the map
                shuffledNeuron := layer.Neurons[neuronIDs[(i+1)%len(neuronIDs)]]
                neuron.Connections = shuffledNeuron.Connections // Assign shuffled connections
                layer.Neurons[neuronID] = neuron // Put the modified neuron back into the map
            }
        }
    }

    return nil
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

