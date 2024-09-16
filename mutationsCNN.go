package dense

import (
    "math/rand"
    "fmt"
)




func AddCNNLayer(config *NetworkConfig) error {
    newLayer := Layer{
        LayerType: "conv",
        Filters: []Filter{
            {
                Weights: Random2DSlice(3, 3), // For example, 3x3 filter
                Bias:    rand.Float64(),
            },
        },
        Stride:  1,
        Padding: 1, // Default padding, adjust as necessary
    }

    config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
    return nil
}


func MutateCNNWeights(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 3 {
        return fmt.Errorf("insufficient arguments for MutateCNNWeights")
    }

    // Typecast the first argument to *NetworkConfig
    config, ok := args[0].(*NetworkConfig)
    if !ok {
        return fmt.Errorf("invalid type for NetworkConfig")
    }

    // Typecast the second argument to float64 for learningRate
    learningRate, ok := args[1].(float64)
    if !ok {
        return fmt.Errorf("invalid type for learningRate")
    }

    // Typecast the third argument to int for mutationRate
    mutationRate, ok := args[2].(int)
    if !ok {
        return fmt.Errorf("invalid type for mutationRate")
    }

    // Iterate through the hidden layers
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

    return nil
}

func MutateCNNBiases(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 3 {
        return fmt.Errorf("insufficient arguments for MutateCNNBiases")
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

    // Iterate through the hidden layers
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "conv" {
            for i := range layer.Filters {
                if rand.Intn(100) < mutationRate {
                    layer.Filters[i].Bias += rand.NormFloat64() * learningRate
                }
            }
        }
    }

    return nil
}


func RandomizeCNNWeights(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for RandomizeCNNWeights")
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

    // Iterate through the hidden layers
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "conv" {
            for i := range layer.Filters {
                if rand.Intn(100) < mutationRate {
                    // Randomize the weights of the filter using Random2DSlice
                    layer.Filters[i].Weights = Random2DSlice(len(layer.Filters[i].Weights), len(layer.Filters[i].Weights[0]))
                }
            }
        }
    }

    return nil
}


func Random2DSlice(rows, cols int) [][]float64 {
    slice := make([][]float64, rows)
    for i := range slice {
        slice[i] = RandomSlice(cols)
    }
    return slice
}

func InvertCNNWeights(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for InvertCNNWeights")
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

    // Iterate through the hidden layers
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "conv" {
            for i := range layer.Filters {
                if rand.Intn(100) < mutationRate {
                    for j := range layer.Filters[i].Weights {
                        for k := range layer.Filters[i].Weights[j] {
                            // Invert the weights of the filter
                            layer.Filters[i].Weights[j][k] = -layer.Filters[i].Weights[j][k]
                        }
                    }
                }
            }
        }
    }

    return nil
}


func AddCNNLayerAtRandomPosition(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for AddCNNLayerAtRandomPosition")
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
            LayerType: "conv",
            Filters: []Filter{
                {
                    Weights: Random2DSlice(3, 3),
                    Bias:    rand.Float64(),
                },
            },
            Stride:  1,
            Padding: 1,
        }

        // Insert at a random position
        pos := rand.Intn(len(config.Layers.Hidden) + 1)
        config.Layers.Hidden = append(config.Layers.Hidden[:pos], append([]Layer{newLayer}, config.Layers.Hidden[pos:]...)...)
    }

    return nil
}
