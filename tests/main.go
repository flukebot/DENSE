package main

import (
	"cortexbuilder"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

func createTestData(envPath string) {
	csvFilePath := envPath + "/data.csv"
	if _, err := os.Stat(csvFilePath); os.IsNotExist(err) {
		file, err := os.Create(csvFilePath)
		if err != nil {
			fmt.Println("error creating file:", err)
			return
		}
		defer file.Close()

		writer := csv.NewWriter(file)

		headers := []string{"input1", "input2", "input3", "output1", "output2", "output3"}
		writer.Write(headers)

		writeData(writer, 10000) // Writes 10,000 rows of data
	}
}

func writeData(writer *csv.Writer, numRows int) {
	for i := 1; i <= numRows; i++ {
		inputs := []string{strconv.Itoa(i), strconv.Itoa(i + 1), strconv.Itoa(i + 2)}
		outputs := []string{strconv.Itoa(2 * i), strconv.Itoa(2 * (i + 1)), strconv.Itoa(2 * (i + 2))}
		data := append(inputs, outputs...)
		writer.Write(data)
	}
	writer.Flush() // Flush the buffer
}

func loadTestData(filePath string) ([]cortexbuilder.TestData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var testData []cortexbuilder.TestData
	for _, record := range records[1:] { // skip headers
		inputs := map[string]float64{
			"input1": parseFloat(record[0]),
			"input2": parseFloat(record[1]),
			"input3": parseFloat(record[2]),
		}
		outputs := map[string]float64{
			"output1": parseFloat(record[3]),
			"output2": parseFloat(record[4]),
			"output3": parseFloat(record[5]),
		}
		testData = append(testData, cortexbuilder.TestData{Inputs: inputs, Outputs: outputs})
	}
	return testData, nil
}

func parseFloat(value string) float64 {
	result, _ := strconv.ParseFloat(value, 64)
	return result
}

func saveNetworkConfig(config *cortexbuilder.NetworkConfig, filename string) error {
	configJSON, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, configJSON, 0644)
}

func logImprovement(iteration int, error float64, improvement float64) error {
	logFile, err := os.OpenFile("improvement_log.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer logFile.Close()

	logEntry := fmt.Sprintf("Iteration %d: Improved Error: %f (Improvement: %.2f%%)\n", iteration, error, improvement)
	_, err = logFile.WriteString(logEntry)
	if err != nil {
		return err
	}

	return nil
}

func evaluateNetwork(config *cortexbuilder.NetworkConfig, testData []cortexbuilder.TestData) float64 {
	totalError := 0.0
	for _, data := range testData {
		outputs := cortexbuilder.Feedforward(config, data.Inputs)
		for key, expected := range data.Outputs {
			actual := outputs[key]
			totalError += math.Abs(expected - actual)
		}
	}
	return totalError
}

// Add new layers or neurons randomly to increase complexity
func mutateNetwork(config *cortexbuilder.NetworkConfig) {
	rand.Seed(time.Now().UnixNano())

	// Randomly add a new neuron in an existing hidden layer
	if rand.Float64() < 0.3 { // 30% chance to add a neuron
		for i := range config.Layers.Hidden {
			newNeuronID := fmt.Sprintf("new_neuron_%d", rand.Intn(1000))
			newNeuron := cortexbuilder.Neuron{
				ActivationType: "relu",
				Connections:    make(map[string]cortexbuilder.Connection),
				Bias:           rand.NormFloat64(),
			}
			for inputID := range config.Layers.Input.Neurons {
				newNeuron.Connections[inputID] = cortexbuilder.Connection{
					Weight: rand.NormFloat64(),
				}
			}
			config.Layers.Hidden[i].Neurons[newNeuronID] = newNeuron
			break
		}
	}

	// Randomly add a new hidden layer
	if rand.Float64() < 0.2 { // 20% chance to add a new hidden layer
		newLayer := cortexbuilder.Layer{
			Neurons: map[string]cortexbuilder.Neuron{},
		}
		for i := 0; i < rand.Intn(3)+1; i++ { // Add 1 to 3 neurons in this new layer
			newNeuronID := fmt.Sprintf("new_hidden_neuron_%d", rand.Intn(1000))
			newNeuron := cortexbuilder.Neuron{
				ActivationType: "relu",
				Connections:    make(map[string]cortexbuilder.Connection),
				Bias:           rand.NormFloat64(),
			}
			// Connect to previous layer's neurons
			if len(config.Layers.Hidden) > 0 {
				for prevNeuronID := range config.Layers.Hidden[len(config.Layers.Hidden)-1].Neurons {
					newNeuron.Connections[prevNeuronID] = cortexbuilder.Connection{
						Weight: rand.NormFloat64(),
					}
				}
			}
			newLayer.Neurons[newNeuronID] = newNeuron
		}
		config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
	}

	// Randomly tweak weights and biases
	for _, layer := range config.Layers.Hidden {
		for nodeID, neuron := range layer.Neurons {
			// Randomly tweak weights
			for inputID := range neuron.Connections {
				neuron.Connections[inputID] = cortexbuilder.Connection{
					Weight: neuron.Connections[inputID].Weight + rand.NormFloat64()*0.1,
				}
			}
			// Randomly tweak biases
			neuron.Bias += rand.NormFloat64() * 0.1
			layer.Neurons[nodeID] = neuron
		}
	}
}

// DeepCopy creates a deep copy of the NetworkConfig to avoid concurrent map access
func deepCopyNetworkConfig(original *cortexbuilder.NetworkConfig) *cortexbuilder.NetworkConfig {
	copyData, _ := json.Marshal(original)
	var copy cortexbuilder.NetworkConfig
	json.Unmarshal(copyData, &copy)
	return &copy
}

func main() {
	// Step 1: Create test data
	createTestData(".")

	// Step 2: Load the test data
	testData, err := loadTestData("./data.csv")
	if err != nil {
		fmt.Println("Error loading test data:", err)
		return
	}

	// Step 3: Create test network configuration
	config := cortexbuilder.CreateTestNetworkConfig()

	// Step 4: Evaluate the initial network performance
	bestError := evaluateNetwork(config, testData)
	fmt.Printf("Initial Error: %f\n", bestError)

	// Step 5: Save the initial network configuration
	err = saveNetworkConfig(config, "initial_network.json")
	if err != nil {
		fmt.Println("Error saving network configuration:", err)
		return
	}

	// Set the improvement threshold to 10%
	targetImprovement := 0.9 * bestError

	// Step 6: Start the hill climbing optimization
	var wg sync.WaitGroup
	mutatedConfigs := make(chan *cortexbuilder.NetworkConfig, 10)
	errors := make(chan float64, 10)

	for i := 0; i < 4; i++ { // Run 4 parallel workers
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < 250; i++ { // Each worker does 250 iterations (total = 1000)
				newConfig := deepCopyNetworkConfig(config) // Use deep copy here
				mutateNetwork(newConfig)
				newError := evaluateNetwork(newConfig, testData)

				mutatedConfigs <- newConfig
				errors <- newError
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(mutatedConfigs)
		close(errors)
	}()

	for i := 0; i < 1000; i++ {
		newConfig := <-mutatedConfigs
		newError := <-errors

		if newError < bestError {
			config = newConfig
			bestError = newError
			improvement := (1 - bestError/targetImprovement) * 100
			fmt.Printf("Iteration %d: Improved Error: %f (Improvement: %.2f%%)\n", i, bestError, improvement)

			// Save the improved network configuration
			err = saveNetworkConfig(config, fmt.Sprintf("network_iteration_%d.json", i))
			if err != nil {
				fmt.Println("Error saving network configuration:", err)
				return
			}

			// Log the improvement
			err = logImprovement(i, bestError, improvement)
			if err != nil {
				fmt.Println("Error logging improvement:", err)
				return
			}

			// Check if we've improved by at least 10%
			if bestError <= targetImprovement {
				fmt.Println("Target improvement achieved!")
				break
			}
		} else {
			fmt.Printf("Iteration %d: No Improvement\n", i)
		}
	}

	fmt.Printf("Final Best Error: %f\n", bestError)
}
