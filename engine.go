package dense

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

type ImageData struct {
	FileName  string             `json:"file_name"`
	Label     int                `json:"label"`
	OutputMap map[string]float64 `json:"output_map"`
}

// TestData is a generalized interface for different types of test data.
type TestData interface {
	GetInputs() map[string]interface{} // Convert data to a format suitable for feeding into the model
	GetLabel() int                     // Get the label or expected output for comparison
}

func GenerateModelsIfNotExist(modelDir string, numModels, inputSize, outputSize int, outputTypes []string, projectName string) error {
	// Create the directory to store the models
	if err := os.MkdirAll(modelDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}

	// Generate and save models
	for i := 0; i < numModels; i++ {
		modelID := fmt.Sprintf("model_%d", i)

		// Define the number of neurons in the first layer
		firstLayerNeurons := 128
		modelConfig := CreateCustomNetworkConfig(inputSize, firstLayerNeurons, outputSize, outputTypes, modelID, projectName)

		// Serialize the model to JSON
		modelFilePath := filepath.Join(modelDir, modelID+".json")
		modelFile, err := os.Create(modelFilePath)
		if err != nil {
			return fmt.Errorf("failed to create model file %s: %w", modelFilePath, err)
		}
		defer modelFile.Close()

		// Encode the model into the JSON file
		encoder := json.NewEncoder(modelFile)
		if err := encoder.Encode(modelConfig); err != nil {
			return fmt.Errorf("failed to serialize model %s: %w", modelFilePath, err)
		}

		log.Printf("Saved model %d to %s\n", i, modelFilePath)
	}

	return nil
}

// SaveLayerStates processes the models in the generation directory and saves layer states for the input data.
func SaveLayerStates(generationDir string, data *[]interface{}, imgDir string) {
	files, err := ioutil.ReadDir(generationDir)
	if err != nil {
		fmt.Printf("Failed to read models directory: %v\n", err)
		return
	}

	// Get the number of available CPU cores and create a semaphore based on this number
	numCores := runtime.NumCPU()
	semaphore := make(chan struct{}, numCores)

	for _, value := range files {

		if filepath.Ext(value.Name()) != ".json" {
			continue // Skip non-JSON files
		}

		// Remove the file extension from the model file name
		modelName := strings.TrimSuffix(value.Name(), filepath.Ext(value.Name()))

		// Generate the full file path for LoadModel
		filePath := filepath.Join(generationDir, value.Name())
		fmt.Println("Processing Model:", modelName)

		// Assuming LoadModel takes the full file path as an argument
		modelConfig, err := LoadModel(filePath)
		if err != nil {
			fmt.Println("Failed to load model:", err)
			return
		}

		layerStateNumber := GetLastHiddenLayerIndex(modelConfig)

		// Construct the shard folder path inside the model's folder
		modelFolderPath := filepath.Join(generationDir, modelName)
		shardFolderPath := filepath.Join(modelFolderPath, fmt.Sprintf("layer_%d_shards", layerStateNumber))

		// Check if the shard folder for this layer already exists
		if _, err := os.Stat(shardFolderPath); !os.IsNotExist(err) {
			fmt.Printf("Shard folder for layer %d already exists in model %s, skipping...\n", layerStateNumber, modelName)
			continue
		}

		// Create the shard folder if it doesn't exist
		err = os.MkdirAll(shardFolderPath, os.ModePerm)
		if err != nil {
			fmt.Printf("Failed to create shard folder in model %s: %v\n", modelName, err)
			continue
		}

		// WaitGroup to wait for all goroutines to finish
		var wg sync.WaitGroup

		// Loop through the data and launch a goroutine for each item
		for _, v := range *data {
			semaphore <- struct{}{} // Acquire a semaphore slot
			wg.Add(1)

			// Launch a goroutine for each data item
			go func(d interface{}) {
				defer wg.Done()
				defer func() { <-semaphore }() // Release semaphore slot when done

				var inputs map[string]interface{}
				var inputID string

				// Handle the type of data
				switch d := d.(type) {
				case ImageData:
					inputs = ConvertImageToInputs(filepath.Join(imgDir, d.FileName)) // Convert image to input values
					inputID = d.FileName
					//fmt.Println(d.Label)
				default:
					fmt.Printf("Unknown data type: %T\n", d)
					return
				}

				// Construct the path to check if the shard for this input already exists in the model's shard folder
				shardFilePath := filepath.Join(shardFolderPath, fmt.Sprintf("input_%s.csv", inputID))

				// Check if the shard already exists for this input
				if _, err := os.Stat(shardFilePath); err == nil {
					fmt.Printf("Shard already exists for input %s, skipping...\n", inputID)
					return
				}

				// Run Feedforward and save the layer state if it doesn't exist
				outputPredicted, layerState := FeedforwardLayerStateSavingShard(modelConfig, inputs, layerStateNumber, filePath)
				_ = outputPredicted
				SaveShardedLayerState(layerState, filePath, layerStateNumber, inputID)

			}(v)
		}

		// Wait for all goroutines to finish
		wg.Wait()
	}

	fmt.Println("All models processed.")
}

func EvaluateModelAccuracyFromLayerState(generationDir string, data *[]interface{}, imgDir string) {
	files, err := ioutil.ReadDir(generationDir)
	if err != nil {
		fmt.Printf("Failed to read models directory: %v\n", err)
		return
	}

	// Get the number of available CPU cores and create a semaphore based on this number
	numCores := runtime.NumCPU()
	semaphore := make(chan struct{}, numCores)

	for _, value := range files {
		if filepath.Ext(value.Name()) != ".json" {
			continue // Skip non-JSON files
		}

		// Remove the file extension from the model file name
		modelName := strings.TrimSuffix(value.Name(), filepath.Ext(value.Name()))

		// Generate the full file path for LoadModel
		modelFilePath := filepath.Join(generationDir, value.Name())
		fmt.Println("Evaluating Model:", modelName)

		// Load the model configuration
		modelConfig, err := LoadModel(modelFilePath)
		if err != nil {
			fmt.Println("Failed to load model:", err)
			return
		}

		// Check if the model has already been evaluated, if so, skip it
		if modelConfig.Metadata.Evaluated {
			fmt.Printf("Model %s has already been evaluated, skipping...\n", modelName)
			continue
		}

		layerStateNumber := GetLastHiddenLayerIndex(modelConfig)

		// Construct the shard folder path inside the model's folder
		modelFolderPath := filepath.Join(generationDir, modelName)
		shardFolderPath := filepath.Join(modelFolderPath, fmt.Sprintf("layer_%d_shards", layerStateNumber))

		// Check if the shard folder for this layer exists
		if _, err := os.Stat(shardFolderPath); os.IsNotExist(err) {
			fmt.Printf("Shard folder for layer %d does not exist in model %s, skipping...\n", layerStateNumber, modelName)
			continue
		}

		// Create the learnedOrNot folder for storing whether the input was correctly predicted or not
		learnedOrNotFolder := CreateLearnedOrNotFolder(modelFilePath, layerStateNumber)

		// WaitGroup to wait for all goroutines to finish
		var wg sync.WaitGroup

		// This will store the results (0 or 1) for each data item
		results := make(chan int, len(*data))

		// Loop through the data and launch a goroutine for each item
		for _, v := range *data {
			semaphore <- struct{}{} // Acquire a semaphore slot
			wg.Add(1)

			go func(d interface{}) {
				defer wg.Done()
				defer func() { <-semaphore }() // Release semaphore slot when done

				var inputID string
				var actualOutput map[string]float64

				// Handle the type of data with type assertion
				switch d := d.(type) {
				case ImageData:
					inputID = d.FileName
					actualOutput = d.OutputMap
				default:
					fmt.Printf("Unknown data type: %T\n", d)
					return
				}

				// Construct the path to the shard file
				shardFilePath := filepath.Join(shardFolderPath, fmt.Sprintf("input_%s.csv", inputID))

				// Check if the shard file exists
				if _, err := os.Stat(shardFilePath); err == nil {

					// Load the saved layer state from the shard file
					savedLayerData := LoadShardedLayerState(modelFilePath, layerStateNumber, inputID)
					if savedLayerData == nil {
						fmt.Printf("No saved layer data for input ID %s. Skipping.\n", inputID)
						return
					}

					// Run the evaluation starting from the saved layer state
					result := ContinueFeedforward(modelConfig, savedLayerData, layerStateNumber)

					// Compare the results and store the prediction status
					isCorrect := CompareOutputs(result, actualOutput)
					if isCorrect {
						results <- 1
					} else {
						results <- 0
					}

					// Save the learned status (true if correct, false if incorrect)
					SaveLearnedOrNot(learnedOrNotFolder, inputID, isCorrect)
				}
			}(v)
		}

		// Wait for all goroutines to finish
		wg.Wait()
		close(results)

		// Tally up the correct results
		totalCorrect := 0
		totalData := len(*data)

		for res := range results {
			totalCorrect += res
		}

		// Print and save the accuracy for this model
		if totalData > 0 {
			accuracy := float64(totalCorrect) / float64(totalData)
			fmt.Printf("Model %s accuracy: %.2f%% (%d/%d)\n", modelName, accuracy*100, totalCorrect, totalData)

			// Save the accuracy and mark the model as evaluated
			modelConfig.Metadata.LastTestAccuracy = accuracy
			modelConfig.Metadata.Evaluated = true // Mark the model as evaluated

			// Save the updated model configuration
			if err := SaveModel(modelFilePath, modelConfig); err != nil {
				fmt.Printf("Failed to save updated model with accuracy: %v\n", err)
			} else {
				fmt.Printf("Updated model %s with accuracy %.2f%% and marked as evaluated.\n", modelName, accuracy*100)
			}
		} else {
			fmt.Println("No data to evaluate accuracy.")
		}
	}

	fmt.Println("All models evaluated.")
}

// Helper function to compare two output maps
func CompareOutputs(predicted, actual map[string]float64) bool {
	if len(predicted) != len(actual) {
		return false
	}

	for key, actualValue := range actual {
		predictedValue, exists := predicted[key]
		if !exists || math.Abs(predictedValue-actualValue) > 1e-6 { // Allowing for floating-point error tolerance
			return false
		}
	}

	return true
}

func EvaluateSingleModelAccuracy(modelConfig *NetworkConfig, data *[]interface{}, layerStateNumber int, generationDir string) (float64, error) {
	modelName := modelConfig.Metadata.ModelID
	modelFolderPath := filepath.Join(generationDir, modelName)
	shardFolderPath := filepath.Join(modelFolderPath, fmt.Sprintf("layer_%d_shards", layerStateNumber))

	// Check if the shard folder for this layer exists
	if _, err := os.Stat(shardFolderPath); os.IsNotExist(err) {
		return 0, fmt.Errorf("Shard folder for layer %d does not exist in model %s", layerStateNumber, modelName)
	}

	// Get the number of available CPU cores and create a semaphore based on this number
	numCores := runtime.NumCPU()
	semaphore := make(chan struct{}, numCores)

	var wg sync.WaitGroup

	// Channel to store the evaluation results (0 for incorrect, 1 for correct)
	results := make(chan int, len(*data))

	// Loop through the data and launch a goroutine for each item
	for _, v := range *data {
		semaphore <- struct{}{} // Acquire a semaphore slot
		wg.Add(1)

		go func(d interface{}) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release semaphore slot when done

			var inputID string
			var actualOutput map[string]float64

			// Handle the type of data with type assertion
			switch d := d.(type) {
			case ImageData:
				inputID = d.FileName
				actualOutput = d.OutputMap
			default:
				fmt.Printf("Unknown data type: %T\n", d)
				return
			}

			// Construct the path to the shard file
			shardFilePath := filepath.Join(shardFolderPath, fmt.Sprintf("input_%s.csv", inputID))

			// Check if the shard file exists
			if _, err := os.Stat(shardFilePath); err == nil {
				// Load the saved layer state from the shard file
				savedLayerData := LoadShardedLayerState(modelFolderPath, layerStateNumber, inputID)
				if savedLayerData == nil {
					fmt.Printf("No saved layer data for input ID %s. Skipping.\n", inputID)
					return
				}

				// Run the evaluation starting from the saved layer state
				result := ContinueFeedforward(modelConfig, savedLayerData, layerStateNumber)
				if CompareOutputs(result, actualOutput) {
					results <- 1
				} else {
					results <- 0
				}
			}
		}(v)
	}

	// Wait for all goroutines to finish
	wg.Wait()
	close(results)

	// Tally up the correct results
	totalCorrect := 0
	totalData := len(*data)

	for res := range results {
		totalCorrect += res
	}

	// Calculate the accuracy
	if totalData > 0 {
		accuracy := float64(totalCorrect) / float64(totalData)
		fmt.Printf("Model %s accuracy: %.2f%% (%d/%d)\n", modelName, accuracy*100, totalCorrect, totalData)
		return accuracy, nil
	} else {
		fmt.Println("No data to evaluate accuracy.")
		return 0, fmt.Errorf("no data to evaluate accuracy")
	}
}

func GenerateChildren(generationDir string, data *[]interface{}, mutationTypes []string, neuronRange [2]int, layerRange [2]int, tries int, allowForTolerance bool, tolerancePercentage float64) {
	fmt.Println("---------Attempting to generate children------------")
	files, err := ioutil.ReadDir(generationDir)
	if err != nil {
		fmt.Printf("Failed to read models directory: %v\n", err)
		return
	}

	// Get the number of available CPU cores and create a semaphore based on this number
	//numCores := runtime.NumCPU()
	//semaphore := make(chan struct{}, numCores)

	for _, value := range files {
		if filepath.Ext(value.Name()) != ".json" {
			continue // Skip non-JSON files
		}

		// Remove the file extension from the model file name
		modelName := strings.TrimSuffix(value.Name(), filepath.Ext(value.Name()))

		// Generate the full file path for LoadModel
		modelFilePath := filepath.Join(generationDir, value.Name())
		fmt.Println("Evaluating Model:", modelName)

		// Load the model configuration
		modelConfig, err := LoadModel(modelFilePath)
		if err != nil {
			fmt.Println("Failed to load model:", err)
			return
		}

		// Check if the model has already been evaluated, if so, skip it
		if modelConfig.Metadata.Evaluated {
			//fmt.Printf("Model %s has already been evaluated, skipping...\n", modelName)
			layerStateNumber := GetLastHiddenLayerIndex(modelConfig)

			// Construct the shard folder path inside the model's folder
			modelFolderPath := filepath.Join(generationDir, modelName)
			shardFolderPath := filepath.Join(modelFolderPath, fmt.Sprintf("layer_%d_shards", layerStateNumber))

			// Check if the shard folder for this layer exists
			if _, err := os.Stat(shardFolderPath); os.IsNotExist(err) {
				fmt.Printf("Shard folder for layer %d does not exist in model %s, skipping...\n", layerStateNumber, modelName)
				continue
			} else {
				modelFilePathFolder := strings.TrimSuffix(modelFilePath, filepath.Ext(modelFilePath))
				highestFolder, err := FindHighestNumberedFolder(modelFilePathFolder, "layer", "learnedornot")
				if err != nil {
					fmt.Println("Error running highest folder")
					continue
				}
				layerOfNotLearned := filepath.Join(modelFilePathFolder, highestFolder)
				lstEvalsTryingToLearn, _ := GetFilesWithExtension(layerOfNotLearned, ".false", 1, false)
				//fmt.Println(lstEvalsTryingToLearn)

				for indexShards, dataShard := range lstEvalsTryingToLearn {
					updated := strings.TrimPrefix(dataShard, "input_")
					fmt.Println(indexShards, dataShard)
					//fmt.Println(modelFilePathFolder + "/layer_" + strconv.Itoa(layerNum) + "_shards/" + dataShard + ".csv")

					//inputIDNumber, _ := ExtractDigitsToInt(dataShard)
					savedLayerData := LoadShardedLayerState(modelFilePath, layerStateNumber, updated)

					// Call the function to get the output map
					outputMap, err := GetOutputByFileName(data, updated)
					if err != nil {
						fmt.Println("Error:", err)
					} else {
						//fmt.Println("Found OutputMap:", outputMap)

						// Perform mutations sequentially and get the matching models
						matchingModels := PerformMutations(generationDir, tries, modelFilePathFolder, savedLayerData, layerStateNumber, mutationTypes, neuronRange, layerRange, outputMap, allowForTolerance, tolerancePercentage)

						// Loop over the results and handle the matching models
						if len(matchingModels) > 0 {
							fmt.Println("Found matching models:")
							for _, model := range matchingModels {
								fmt.Println(model)
							}
						} else {
							fmt.Println("No matching models found.")
						}

						/*if tries <= 0 {
							tries = 100 // Default value for tries if not provided
						}

						// Loop over the number of tries to apply mutation and compare
						for i := 0; i < tries; i++ {
							modelConfig, err := LoadModel(modelFilePathFolder + ".json")
							if err != nil {
								fmt.Println("Failed to load model:", err)
								continue
							}

							// Apply mutation
							mutatedModel := ApplySingleMutation(modelConfig, mutationTypes, neuronRange, layerRange)

							// Continue the feedforward process with the mutated model
							mutatedResult := ContinueFeedforward(mutatedModel, savedLayerData, layerStateNumber)

							// Compare the results with the expected output map
							if FlexibleCompareOutputs(mutatedResult, outputMap, allowForTolerance, tolerancePercentage) {
								fmt.Printf("Try %d: Found matching output!\n", i+1)
							} else {
								fmt.Printf("Try %d: Output does not match.\n", i+1)
							}
						}*/

					}

				}
			}

		}

	}

	fmt.Println("All models evaluated.")
}

// Single-threaded version of the mutation and comparison loop
func PerformMutationsSingleThreaded(generationDir string, tries int, modelFilePathFolder string, savedLayerData interface{}, layerStateNumber int, mutationTypes []string, neuronRange [2]int, layerRange [2]int, outputMap map[string]float64, allowForTolerance bool, tolerancePercentage float64) []*NetworkConfig {
	if tries <= 0 {
		tries = 100 // Default value for tries if not provided
	}

	// Slice to collect results
	var matchingModels []*NetworkConfig

	// Perform mutations sequentially for each try
	for i := 0; i < tries; i++ {
		// Load model
		modelConfig, err := LoadModel(modelFilePathFolder + ".json")
		if err != nil {
			fmt.Println("Failed to load model:", err)
			continue
		}

		// Apply mutation
		mutatedModel := ApplySingleMutation(modelConfig, mutationTypes, neuronRange, layerRange)

		// Continue the feedforward process with the mutated model
		mutatedResult := ContinueFeedforward(mutatedModel, savedLayerData, layerStateNumber)

		// Compare the results with the expected output map
		if FlexibleCompareOutputs(mutatedResult, outputMap, allowForTolerance, tolerancePercentage) {
			//fmt.Printf("Try %d: Found matching output!\n", i+1)
			matchingModels = append(matchingModels, mutatedModel) // Collect the matching model
		} else {
			//fmt.Printf("Try %d: Output does not match.\n", i+1)
		}
	}

	return matchingModels
}

// Multithreaded version of the mutation and comparison loop
func PerformMutations(generationDir string, tries int, modelFilePathFolder string, savedLayerData interface{}, layerStateNumber int, mutationTypes []string, neuronRange [2]int, layerRange [2]int, outputMap map[string]float64, allowForTolerance bool, tolerancePercentage float64) []*NetworkConfig {
	if tries <= 0 {
		tries = 100 // Default value for tries if not provided
	}

	// Channel to collect results
	results := make(chan *NetworkConfig, tries)
	var wg sync.WaitGroup

	// Launch goroutines for each try
	for i := 0; i < tries; i++ {
		wg.Add(1)
		go func(try int) {
			defer wg.Done()

			// Load model
			modelConfig, err := LoadModel(modelFilePathFolder + ".json")
			if err != nil {
				fmt.Println("Failed to load model:", err)
				results <- nil // Send nil if there's an error
				return
			}

			// Apply mutation
			mutatedModel := ApplySingleMutation(modelConfig, mutationTypes, neuronRange, layerRange)

			// Continue the feedforward process with the mutated model
			mutatedResult := ContinueFeedforward(mutatedModel, savedLayerData, layerStateNumber)

			// Compare the results with the expected output map
			if FlexibleCompareOutputs(mutatedResult, outputMap, allowForTolerance, tolerancePercentage) {
				//fmt.Printf("Try %d: Found matching output!\n", try+1)
				results <- mutatedModel // Send the mutated model if there's a match
			} else {
				//fmt.Printf("Try %d: Output does not match.\n", try+1)
				results <- nil // Send nil if there's no match
			}
		}(i)
	}

	// Wait for all goroutines to finish
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect non-nil results (i.e., matching models)
	var matchingModels []*NetworkConfig
	for result := range results {
		if result != nil {
			matchingModels = append(matchingModels, result)
		}
	}

	return matchingModels
}

func ApplySingleMutation(modelConfig *NetworkConfig, mutationTypes []string, neuronRange [2]int, layerRange [2]int) *NetworkConfig {
	// Randomly select a mutation type from the list of mutationTypes
	mutationType := mutationTypes[rand.Intn(len(mutationTypes))]

	// Apply the mutation based on the selected type
	switch mutationType {
	case "AppendNewLayer":
		// Randomize the number of neurons or filters between the provided range
		numNewNeuronsOrFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		AppendNewLayerFullConnections(modelConfig, numNewNeuronsOrFilters)

	case "AppendMultipleLayers":
		// Randomize the number of layers between the provided range
		numNewLayers := rand.Intn(layerRange[1]-layerRange[0]+1) + layerRange[0]
		// Randomize the number of neurons or filters between the provided range
		numNewNeuronsOrFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		AppendMultipleLayers(modelConfig, numNewLayers, numNewNeuronsOrFilters)

	case "AddCNNLayer":
		numNewNeuronsOrFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		AddCNNLayerAtRandomPosition(modelConfig, numNewNeuronsOrFilters)

	case "AddMultipleCNNLayers":
		numNewLayers := rand.Intn(layerRange[1]-layerRange[0]+1) + layerRange[0]
		AddMultipleCNNLayers(modelConfig, 100, numNewLayers)

	case "AddLSTMLayer":
		AddLSTMLayerAtRandomPosition(modelConfig, 100)

	default:
		fmt.Println("Unknown mutation type:", mutationType)
	}

	// Get the previous output activation types before reattaching the output layer
	previousOutputActivationTypes := GetPreviousOutputActivationTypes(modelConfig)

	// Reattach the output layer with the previous activation types
	ReattachOutputLayer(modelConfig, len(previousOutputActivationTypes), previousOutputActivationTypes)

	// Return the mutated model
	return modelConfig
}

// Function to find the output map based on a string input and return the OutputMap
func GetOutputByFileName(data *[]interface{}, fileName string) (map[string]float64, error) {
	for _, item := range *data {
		// Type assertion to check if the item is of type ImageData
		if imageData, ok := item.(ImageData); ok {
			if imageData.FileName == fileName {
				return imageData.OutputMap, nil
			}
		} else {
			return nil, fmt.Errorf("unexpected struct type in data")
		}
	}

	return nil, fmt.Errorf("file with name %s not found", fileName)
}

// Strict comparison function (without tolerance)
func CompareOutputsStrict(result map[string]float64, expectedOutput map[string]float64) bool {
	for key, value := range expectedOutput {
		if resultVal, ok := result[key]; ok {
			// Compare values exactly
			if resultVal != value {
				return false
			}
		} else {
			return false
		}
	}
	return true
}

// Comparison function with tolerance
func CompareOutputsWithTolerance(result map[string]float64, expectedOutput map[string]float64, tolerancePercentage float64) bool {
	matches := 0
	total := len(expectedOutput)

	for key, expectedValue := range expectedOutput {
		if resultValue, ok := result[key]; ok {
			// Calculate the absolute difference and check if it's within tolerance
			diff := math.Abs(resultValue - expectedValue)
			tolerance := (tolerancePercentage / 100) * math.Abs(expectedValue)

			if diff <= tolerance {
				matches++
			}
		}
	}

	// Calculate percentage of matching outputs
	matchPercentage := (float64(matches) / float64(total)) * 100

	// Return true if the matching percentage is equal to or above the tolerance percentage
	return matchPercentage >= tolerancePercentage
}

// Main CompareOutputs function that decides whether to use strict or tolerance-based comparison
func FlexibleCompareOutputs(result map[string]float64, expectedOutput map[string]float64, allowForTolerance bool, tolerancePercentage float64) bool {
	if allowForTolerance {
		return CompareOutputsWithTolerance(result, expectedOutput, tolerancePercentage)
	}
	return CompareOutputsStrict(result, expectedOutput)
}
