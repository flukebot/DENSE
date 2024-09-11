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
)

// MutateNetwork applies a random mutation based on the mutation type
func MutateNetwork(config *NetworkConfig, learningRate float64, mutationRate int) {
    rand.Seed(time.Now().UnixNano())

    // Randomly select the mutation type to apply
    switch rand.Intn(14) { // Updated to include all 14 mutation types
    case int(MutateWeight):
       //  fmt.Println("Applying weight mutation")
        MutateWeights(config, learningRate, mutationRate)
    case int(AddNeuronMutation):
       //  fmt.Println("Adding a neuron")
        AddNeuron(config, mutationRate)
    case int(AddLayerFullConnectionMutation):
       //  fmt.Println("Adding a new fully connected layer")
        AddLayerFullConnections(config, mutationRate)
    case int(AddLayerSparseMutation):
       //  fmt.Println("Adding a new sparse layer")
        AddLayer(config, mutationRate)
    case int(AddLayerRandomPositionMutation):
       //  fmt.Println("Adding a new layer at a random position")
        AddLayerRandomPosition(config, mutationRate)
    case int(MutateActivationFunction):
       //  fmt.Println("Mutating activation functions")
        MutateActivationFunctions(config, mutationRate)
    case int(RemoveNeuronMutation):
       //  fmt.Println("Removing a neuron")
        RemoveNeuron(config, mutationRate)
    case int(RemoveLayerMutation):
       //  fmt.Println("Removing a layer")
        RemoveLayer(config, mutationRate)
    case int(DuplicateNeuronMutation):
       //  fmt.Println("Duplicating a neuron")
        DuplicateNeuron(config, mutationRate)
    case int(MutateBiasMutation):
       //  fmt.Println("Mutating biases")
        MutateBiases(config, mutationRate, learningRate)
    case int(RandomizeWeightsMutation):
       //  fmt.Println("Randomizing weights")
        RandomizeWeights(config, mutationRate)
    case int(SplitNeuronMutation):
       //  fmt.Println("Splitting a neuron")
        SplitNeuron(config, mutationRate)
    case int(SwapLayerActivationsMutation):
       //  fmt.Println("Swapping layer activations")
        SwapLayerActivations(config, mutationRate)
    case int(ShuffleLayerConnectionsMutation):
       //  fmt.Println("Shuffling layer connections")
        ShuffleLayerConnections(config, mutationRate)
    }
}



// MutateWeights randomly mutates the network's weights with a given mutation rate
func MutateWeights(config *NetworkConfig, learningRate float64, mutationRate int) {
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

// AddNeuron adds a new neuron to a random hidden layer based on the mutation rate
func AddNeuron(config *NetworkConfig, mutationRate int) {
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
func AddLayer(config *NetworkConfig, mutationRate int) {
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
func MutateActivationFunctions(config *NetworkConfig, mutationRate int) {
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
            for neuronID, neuron1 := range layer1.Neurons {
                neuron2 := layer2.Neurons[neuronID]
                neuron1.ActivationType, neuron2.ActivationType = neuron2.ActivationType, neuron1.ActivationType
                layer1.Neurons[neuronID], layer2.Neurons[neuronID] = neuron1, neuron2
            }
           //  fmt.Printf("Swapped activation functions between layer %d and layer %d\n", idx1+1, idx2+1)
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

