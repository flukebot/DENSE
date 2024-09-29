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


func saveLayerDataToCSV(data interface{}, modelFilePath string, layerIndex int) {
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
        var record []string
        for key, value := range v {
            record = append(record, key, strconv.FormatFloat(value, 'f', 6, 64))
        }
        writer.Write(record)

    case [][]float64:
        for _, row := range v {
            var record []string
            for _, value := range row {
                record = append(record, strconv.FormatFloat(value, 'f', 6, 64))
            }
            writer.Write(record)
        }

    case [][][]float64:
        for _, image := range v {
            for _, row := range image {
                var record []string
                for _, value := range row {
                    record = append(record, strconv.FormatFloat(value, 'f', 6, 64))
                }
                writer.Write(record)
            }
        }
    }
}

