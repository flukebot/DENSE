package dense

import (
    "fmt"
    "log"
	"os"
	"path/filepath"
	"encoding/json"
	"sync"
	"io/ioutil"
	"runtime"
	"strings"
    "math"
)

type ImageData struct {
	FileName string `json:"file_name"`
	Label    int    `json:"label"`
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
                    fmt.Println(d.Label)
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
            modelConfig.Metadata.Evaluated = true  // Mark the model as evaluated

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
