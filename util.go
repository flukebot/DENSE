package dense

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"runtime"
	"fmt"
	//"syscall/js"
	"crypto/md5"
	"encoding/hex"
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


// Alias for SaveNetworkToFile to match the expected function signature
func SaveNetworkConfig(config *NetworkConfig, filename string) error {
    return SaveNetworkToFile(config, filename)
}


// Function to compute MD5 hash of a JSON serialized object
func computeMD5Hash(v interface{}) (string, error) {
	// Convert the model configuration to a JSON string
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("failed to serialize object to JSON: %v", err)
	}

	// Compute MD5 hash
	hash := md5.Sum(jsonBytes)
	return hex.EncodeToString(hash[:]), nil
}