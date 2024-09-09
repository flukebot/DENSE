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
    AddLayerMutation
)

func MutateNetwork(config *NetworkConfig, learningRate float64, mutationRate int) {
    rand.Seed(time.Now().UnixNano())

    config.mutex.Lock() // Lock for writing
    defer config.mutex.Unlock() // Unlock after mutation

    // Randomly select the mutation type to apply
    switch rand.Intn(3) {
    case int(MutateWeight):
        fmt.Println("Applying weight mutation")
        MutateWeights(config, learningRate, mutationRate)
    case int(AddNeuronMutation):
        fmt.Println("Adding a neuron")
        AddNeuron(config, mutationRate)
    case int(AddLayerMutation):
        fmt.Println("Adding a new layer")
        AddLayer(config, mutationRate)
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
        fmt.Println("No hidden layers found. Adding a new layer first.")
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

        fmt.Printf("Added a new neuron to hidden layer %d\n", layerIdx+1)
    }
}

func AddLayer(config *NetworkConfig, mutationRate int) {
    // Ensure mutationRate is within bounds
    if mutationRate < 0 {
        mutationRate = 0
    } else if mutationRate > 100 {
        mutationRate = 100
    }

    // Randomly decide if we should add a layer based on the mutation rate
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

            // Connect the new neuron to the previous layer
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

        // Connect the new layer to the output layer
        for outputNeuronID, outputNeuron := range config.Layers.Output.Neurons {
            newConnections := make(map[string]Connection)
            for newNeuronID := range newLayer.Neurons {
                newConnections[newNeuronID] = Connection{Weight: rand.NormFloat64()}
            }
            outputNeuron.Connections = newConnections
            config.Layers.Output.Neurons[outputNeuronID] = outputNeuron
        }

        config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
        fmt.Printf("Added a new hidden layer with %d neurons\n", numNewNeurons)
    }
}

// Helper function to choose a random activation function
func randomActivationType() string {
    activationTypes := []string{"relu", "sigmoid", "tanh", "leaky_relu"}
    return activationTypes[rand.Intn(len(activationTypes))]
}