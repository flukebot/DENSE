package dense

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"encoding/csv"
	"path/filepath"
    "strconv"
	"strings"
	//"syscall/js"
)


// EnvType represents the environment type
type EnvType string

const (
	WASM   EnvType = "WASM"
	Linux  EnvType = "Linux"
	MacOS  EnvType = "MacOS"
	Windows EnvType = "Windows"
	Unknown EnvType = "Unknown"
)

// GetEnvType returns the type of environment the program is running on
func GetEnvType() EnvType {
	if runtime.GOARCH == "wasm" && runtime.GOOS == "js" {
		return WASM
	}

	switch runtime.GOOS {
	case "linux":
		return Linux
	case "darwin":
		return MacOS
	case "windows":
		return Windows
	default:
		return Unknown
	}
}

// LoadNetworkConfig loads the neural network configuration from a JSON file.
func LoadNetworkConfig(filename string) (*NetworkConfig, error) {
	// Read the file
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// Unmarshal JSON into the NetworkConfig struct
	var networkConfig NetworkConfig
	err = json.Unmarshal(data, &networkConfig)
	if err != nil {
		return nil, err
	}

	return &networkConfig, nil
}


// Load a network configuration from a JSON file
func LoadNetworkFromFile(filename string) (*NetworkConfig, error) {
	data, err := readFromFile(filename)
	if err != nil {
		return nil, err
	}
	var config NetworkConfig
	err = json.Unmarshal(data, &config)
	return &config, err
}


// Helper functions for file IO
func writeToFile(filename string, data []byte) error {
	return os.WriteFile(filename, data, 0644)
}

func readFromFile(filename string) ([]byte, error) {
	return os.ReadFile(filename)
}


// Save the network configuration as a JSON file
func SaveNetworkToFile(config *NetworkConfig, filename string) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	err = writeToFile(filename, data)
	return err
}

// Check if a directory exists
func CheckDirExists(dirPath string) bool {
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		return false // Directory does not exist
	}
	return true // Directory exists
}

// createDirectory creates a directory if it doesn't already exist.
func CreateDirectory(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			return fmt.Errorf("failed to create directory: %v", err)
		}
	}
	return nil
}

func saveLayerDataToCSV(data interface{}, modelFilePath string, layerIndex int, inputID string) {
    // Extract the directory and the model file name without the extension
    dir, file := filepath.Split(modelFilePath)
    modelName := strings.TrimSuffix(file, filepath.Ext(file))
    
    // Create the folder path where the CSV file will be saved
    folderPath := filepath.Join(dir, modelName)
    os.MkdirAll(folderPath, os.ModePerm)
    
    // Define the CSV file name using the layer index
    fileName := filepath.Join(folderPath, "layer_" + strconv.Itoa(layerIndex) + ".csv")

    // Open the CSV file in append mode. If the file does not exist, it will be created.
    fileHandle, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        panic(err)
    }
    defer fileHandle.Close()

    writer := csv.NewWriter(fileHandle)
    defer writer.Flush()

    // Write the data to the CSV file depending on its type
    switch v := data.(type) {
    case map[string]float64:
        for key, value := range v {
            record := []string{inputID, key, strconv.FormatFloat(value, 'g', -1, 64)}
            writer.Write(record)
        }

    case [][]float64:
        for _, row := range v {
            var record []string
            record = append(record, inputID) // Include inputID at the beginning
            for _, value := range row {
                record = append(record, strconv.FormatFloat(value, 'g', -1, 64))
            }
            writer.Write(record)
        }

    case [][][]float64:
        for _, image := range v {
            for _, row := range image {
                var record []string
                record = append(record, inputID) // Include inputID at the beginning
                for _, value := range row {
                    record = append(record, strconv.FormatFloat(value, 'g', -1, 64))
                }
                writer.Write(record)
            }
        }
    }
}






// LoadCSVLayerState loads the saved layer state for a specific inputID from a CSV file and returns it as a suitable data structure.
func LoadCSVLayerState(filePath string, inputID string) interface{} {
    // Open the CSV file
    file, err := os.Open(filePath)
    if err != nil {
        panic(err)
    }
    defer file.Close()

    // Create a new CSV reader
    reader := csv.NewReader(file)

    // Read all the data from the CSV file
    records, err := reader.ReadAll()
    if err != nil {
        panic(err)
    }

    // Assuming the saved layer state is a map[string]float64, adapt as needed based on your data structure
    savedLayerState := make(map[string]float64)

    // Populate the map with the data from the CSV file corresponding to the inputID
    for _, record := range records {
        if len(record) >= 3 { // Assuming each record has inputID, key, and value
            recordInputID := record[0]
            if recordInputID != inputID {
                continue // Skip records not matching the inputID
            }
            key := record[1]
            value, err := strconv.ParseFloat(record[2], 64)
            if err != nil {
                panic(err)
            }
            savedLayerState[key] = value
        }
    }

    return savedLayerState
}



// SaveShardedLayerState saves the layer state for each input as a separate file (shard) in a dedicated folder.
func SaveShardedLayerState(data interface{}, modelFilePath string, layerIndex int, inputID string) {
    // Create a folder for storing shards for the given layer
    dir, file := filepath.Split(modelFilePath)
    modelName := strings.TrimSuffix(file, filepath.Ext(file))
    layerShardFolder := filepath.Join(dir, modelName, fmt.Sprintf("layer_%d_shards", layerIndex))
    
    // Create the directory if it doesn't exist
    os.MkdirAll(layerShardFolder, os.ModePerm)

    // Define the CSV file name for the specific input ID shard
    fileName := filepath.Join(layerShardFolder, fmt.Sprintf("input_%s.csv", inputID))

    // Save the state for this input to a CSV file
    fileHandle, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        panic(err)
    }
    defer fileHandle.Close()

    writer := csv.NewWriter(fileHandle)

    // Write data to the CSV file based on the type of `data`
    switch v := data.(type) {
    case map[string]float64:
        for key, value := range v {
            record := []string{inputID, key, strconv.FormatFloat(value, 'g', -1, 64)}
            writer.Write(record)
        }

    case [][]float64:
        for _, row := range v {
            var record []string
            record = append(record, inputID)
            for _, value := range row {
                record = append(record, strconv.FormatFloat(value, 'g', -1, 64))
            }
            writer.Write(record)
        }

    case [][][]float64:
        for _, image := range v {
            for _, row := range image {
                var record []string
                record = append(record, inputID)
                for _, value := range row {
                    record = append(record, strconv.FormatFloat(value, 'g', -1, 64))
                }
                writer.Write(record)
            }
        }
    }

    // Flush the writer to ensure everything is written properly, including newlines.
    writer.Flush()

    // Check for any error that occurred during writing or flushing
    if err := writer.Error(); err != nil {
        panic(err)
    }
}



// LoadShardedLayerState loads the saved layer state for a specific inputID shard from a CSV file.
func LoadShardedLayerState(modelFilePath string, layerIndex int, inputID string) interface{} {
    // Define the path to the shard for this input
    dir, fileName := filepath.Split(modelFilePath)
    modelName := strings.TrimSuffix(fileName, filepath.Ext(fileName))
    layerShardFolder := filepath.Join(dir, modelName, fmt.Sprintf("layer_%d_shards", layerIndex))
    shardFilePath := filepath.Join(layerShardFolder, fmt.Sprintf("input_%s.csv", inputID))
    //fmt.Println("shard",shardFilePath)
    // Open the shard file
    fileHandle, err := os.Open(shardFilePath)
    if err != nil {
        panic(err)
    }
    defer fileHandle.Close()

    reader := csv.NewReader(fileHandle)
    records, err := reader.ReadAll()
    if err != nil {
        panic(err)
    }

    // Assuming the saved layer state is a map[string]float64
    savedLayerState := make(map[string]float64)
    for _, record := range records {
        if len(record) >= 3 && record[0] == inputID {
            key := record[1]
            value, err := strconv.ParseFloat(record[2], 64)
            if err != nil {
                panic(err)
            }
            savedLayerState[key] = value
        }
    }

    return savedLayerState
}



// CreateLearnedOrNotFolder creates the folder for learned or not tracking inside the model folder
func CreateLearnedOrNotFolder(modelFilePath string, layerIndex int) string {
    // Extract the directory and the model name without the extension
    dir, file := filepath.Split(modelFilePath)
    modelName := strings.TrimSuffix(file, filepath.Ext(file))

    // Create the folder path where the `learnedOrNot` files will be saved
    learnedOrNotFolder := filepath.Join(dir, modelName, fmt.Sprintf("layer_%d_learnedornot", layerIndex))

    // Create the directory if it doesn't exist
    err := os.MkdirAll(learnedOrNotFolder, os.ModePerm)
    if err != nil {
        panic(fmt.Sprintf("Failed to create learnedOrNot folder: %v", err))
    }

    return learnedOrNotFolder
}

// SaveLearnedOrNot saves whether the input was correctly predicted (true/false) in the `learnedOrNot` folder
func SaveLearnedOrNot(learnedOrNotFolder string, inputID string, isCorrect bool) {
    var learnedFileName string

    if isCorrect {
        learnedFileName = fmt.Sprintf("input_%s_.true", inputID)
    } else {
        learnedFileName = fmt.Sprintf("input_%s_.false", inputID)
    }

    // Path for the learned or not file
    filePath := filepath.Join(learnedOrNotFolder, learnedFileName)

    // Write the result (true/false) to the file
    content := strconv.FormatBool(isCorrect)
    err := os.WriteFile(filePath, []byte(content), 0644)
    if err != nil {
        panic(fmt.Sprintf("Failed to write file %s: %v", filePath, err))
    }
}