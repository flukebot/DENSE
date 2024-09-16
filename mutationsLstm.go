package dense

import (
    "math/rand"
    "fmt"
)



func AddLSTMLayer(config *NetworkConfig) error {
    newLayer := Layer{
        LayerType: "lstm",
        LSTMCells: []LSTMCell{
            {
                InputWeights:  RandomSlice(10),
                ForgetWeights: RandomSlice(10),
                OutputWeights: RandomSlice(10),
                CellWeights:   RandomSlice(10),
                Bias:          rand.Float64(),
            },
        },
    }

    // Append the new LSTM layer
    config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
    return nil
}


func MutateLSTMWeights(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 3 {
        return fmt.Errorf("insufficient arguments for MutateLSTMWeights")
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

    // Iterate through LSTM layers and mutate weights
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "lstm" {
            for i := range layer.LSTMCells {
                if rand.Intn(100) < mutationRate {
                    for j := range layer.LSTMCells[i].InputWeights {
                        layer.LSTMCells[i].InputWeights[j] += rand.NormFloat64() * learningRate
                    }
                    for j := range layer.LSTMCells[i].ForgetWeights {
                        layer.LSTMCells[i].ForgetWeights[j] += rand.NormFloat64() * learningRate
                    }
                    for j := range layer.LSTMCells[i].OutputWeights {
                        layer.LSTMCells[i].OutputWeights[j] += rand.NormFloat64() * learningRate
                    }
                    for j := range layer.LSTMCells[i].CellWeights {
                        layer.LSTMCells[i].CellWeights[j] += rand.NormFloat64() * learningRate
                    }
                }
            }
        }
    }

    return nil
}


func MutateLSTMBiases(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 3 {
        return fmt.Errorf("insufficient arguments for MutateLSTMBiases")
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

    // Iterate through LSTM layers and mutate biases
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "lstm" {
            for i := range layer.LSTMCells {
                if rand.Intn(100) < mutationRate {
                    layer.LSTMCells[i].Bias += rand.NormFloat64() * learningRate
                }
            }
        }
    }

    return nil
}



func RandomizeLSTMWeights(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for RandomizeLSTMWeights")
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

    // Iterate through LSTM layers and randomize weights
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "lstm" {
            for i := range layer.LSTMCells {
                if rand.Intn(100) < mutationRate {
                    layer.LSTMCells[i].InputWeights = RandomSlice(len(layer.LSTMCells[i].InputWeights))
                    layer.LSTMCells[i].ForgetWeights = RandomSlice(len(layer.LSTMCells[i].ForgetWeights))
                    layer.LSTMCells[i].OutputWeights = RandomSlice(len(layer.LSTMCells[i].OutputWeights))
                    layer.LSTMCells[i].CellWeights = RandomSlice(len(layer.LSTMCells[i].CellWeights))
                }
            }
        }
    }

    return nil
}


func InvertLSTMWeights(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for InvertLSTMWeights")
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

    // Iterate through LSTM layers and invert weights
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "lstm" {
            for i := range layer.LSTMCells {
                if rand.Intn(100) < mutationRate {
                    for j := range layer.LSTMCells[i].InputWeights {
                        layer.LSTMCells[i].InputWeights[j] = -layer.LSTMCells[i].InputWeights[j]
                    }
                    for j := range layer.LSTMCells[i].ForgetWeights {
                        layer.LSTMCells[i].ForgetWeights[j] = -layer.LSTMCells[i].ForgetWeights[j]
                    }
                    for j := range layer.LSTMCells[i].OutputWeights {
                        layer.LSTMCells[i].OutputWeights[j] = -layer.LSTMCells[i].OutputWeights[j]
                    }
                    for j := range layer.LSTMCells[i].CellWeights {
                        layer.LSTMCells[i].CellWeights[j] = -layer.LSTMCells[i].CellWeights[j]
                    }
                }
            }
        }
    }

    return nil
}

func AddLSTMLayerAtRandomPosition(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for AddLSTMLayerAtRandomPosition")
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
            LayerType: "lstm",
            LSTMCells: []LSTMCell{
                {
                    InputWeights:  RandomSlice(10),
                    ForgetWeights: RandomSlice(10),
                    OutputWeights: RandomSlice(10),
                    CellWeights:   RandomSlice(10),
                    Bias:          rand.Float64(),
                },
            },
        }

        // Insert at a random position
        pos := rand.Intn(len(config.Layers.Hidden) + 1)
        config.Layers.Hidden = append(config.Layers.Hidden[:pos], append([]Layer{newLayer}, config.Layers.Hidden[pos:]...)...)
    }

    return nil
}



func MutateLSTMCells(args ...interface{}) error {
    // Check for sufficient arguments
    if len(args) < 2 {
        return fmt.Errorf("insufficient arguments for MutateLSTMCells")
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

    // Loop through all layers
    for _, layer := range config.Layers.Hidden {
        // Only target LSTM layers
        if layer.LayerType == "lstm" {
            // Iterate through each LSTM cell in the layer
            for i, cell := range layer.LSTMCells {
                // Mutate the input, forget, output, and cell weights
                for j := range cell.InputWeights {
                    if rand.Intn(100) < mutationRate {
                        cell.InputWeights[j] += rand.NormFloat64()
                    }
                }
                for j := range cell.ForgetWeights {
                    if rand.Intn(100) < mutationRate {
                        cell.ForgetWeights[j] += rand.NormFloat64()
                    }
                }
                for j := range cell.OutputWeights {
                    if rand.Intn(100) < mutationRate {
                        cell.OutputWeights[j] += rand.NormFloat64()
                    }
                }
                for j := range cell.CellWeights {
                    if rand.Intn(100) < mutationRate {
                        cell.CellWeights[j] += rand.NormFloat64()
                    }
                }

                // Mutate the biases
                if rand.Intn(100) < mutationRate {
                    cell.Bias += rand.NormFloat64()
                }

                // Update the mutated LSTM cell in the layer
                layer.LSTMCells[i] = cell
            }
        }
    }

    return nil
}
