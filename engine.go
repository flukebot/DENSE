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
)

type ImageData struct {
	FileName string `json:"file_name"`
	Label    int    `json:"label"`
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

        // Generate the full file path for LoadModel
        filePath := filepath.Join(generationDir, value.Name())
		fmt.Println("Model", value.Name())
        // Assuming LoadModel takes the full file path as an argument
        modelConfig, err := LoadModel(filePath)
        if err != nil {
            fmt.Println("Failed to load model:", err)
            return
        }

        layerStateNumber := GetLastHiddenLayerIndex(modelConfig)

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
                default:
                    fmt.Printf("Unknown data type: %T\n", d)
                    return
                }

                // Construct the path to check if the shard for this input already exists
                shardFilePath := filepath.Join(filepath.Dir(filePath), fmt.Sprintf("layer_%d_shards", layerStateNumber), fmt.Sprintf("input_%s.csv", inputID))

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





