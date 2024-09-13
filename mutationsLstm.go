package dense

import "math/rand"


func MutateLSTMWeights(config *NetworkConfig, learningRate float64, mutationRate int) {
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
}

func MutateLSTMBiases(config *NetworkConfig, mutationRate int, learningRate float64) {
    for _, layer := range config.Layers.Hidden {
        if layer.LayerType == "lstm" {
            for i := range layer.LSTMCells {
                if rand.Intn(100) < mutationRate {
                    layer.LSTMCells[i].Bias += rand.NormFloat64() * learningRate
                }
            }
        }
    }
}


func RandomizeLSTMWeights(config *NetworkConfig, mutationRate int) {
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
}

func InvertLSTMWeights(config *NetworkConfig, mutationRate int) {
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
}

func AddLSTMLayerAtRandomPosition(config *NetworkConfig, mutationRate int) {
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
}
