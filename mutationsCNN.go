package dense

import "math/rand"


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
                    layer.Filters[i].Weights = random2DSlice(len(layer.Filters[i].Weights), len(layer.Filters[i].Weights[0]))
                }
            }
        }
    }
}

func random2DSlice(rows, cols int) [][]float64 {
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

func AddCNNLayerAtRandomPosition(config *NetworkConfig, mutationRate int) {
    if rand.Intn(100) < mutationRate {
        newLayer := Layer{
            LayerType: "conv",
            Filters: []Filter{
                {
                    Weights: random2DSlice(3, 3),
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
}
