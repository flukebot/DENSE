package dense

import (
	"math/rand"
	"time"
)

// Hill climbing mutation for network weights
func MutateWeights(config *NetworkConfig, learningRate float64) {
	rand.Seed(time.Now().UnixNano())
	for _, layer := range config.Layers.Hidden {
		for _, neuron := range layer.Neurons {
			for connID := range neuron.Connections {
				neuron.Connections[connID] = Connection{Weight: neuron.Connections[connID].Weight + rand.NormFloat64()*learningRate}
			}
			neuron.Bias += rand.NormFloat64() * learningRate
		}
	}
}