package main

import (
	"dense" // Import your dense package
	"flag"
	"fmt"
	"os"
)

func main() {
    // If no arguments are passed, do nothing and return
    if len(os.Args) == 1 {
        fmt.Println("No commands passed. Use -help for available commands.")
        return
    }

    // Setup command-line flags
    createModelCmd := flag.Bool("createModel", false, "Create a new neural network model")
    runModelCmd := flag.Bool("runModel", false, "Run a neural network model with a JSON config")
    jsonFilePath := flag.String("json", "", "Path to the JSON model config file")

    // Parse the command-line flags
    flag.Parse()

    // Check which command was passed and handle accordingly
    if *createModelCmd {
        handleCreateModel()
    } else if *runModelCmd {
        handleRunModel(*jsonFilePath)
    } else {
        fmt.Println("Invalid command. Use -help for available commands.")
    }
}

// Function to handle creating a new model
func handleCreateModel() {
    // This function generates a random model
    model := dense.RandomizeNetworkStaticTesting()
    fmt.Println("Created Neural Network Model:\n", model)
}

// Function to handle running a neural network model from JSON
func handleRunModel(jsonPath string) {
    if jsonPath == "" {
        fmt.Println("Please provide the path to the JSON config using -json flag")
        return
    }

    // Load the network config from JSON
    config, err := dense.LoadNetworkConfig(jsonPath)
    if err != nil {
        fmt.Println("Error loading network config:", err)
        return
    }

    // Sample input values (you can modify or pass this dynamically)
    inputValues := map[string]float64{
        "input1": 1.0,
        "input2": 0.5,
        "input3": 0.2,
    }

    // Run the feedforward computation
    outputs := dense.Feedforward(config, inputValues)

    // Print the output
    fmt.Println("Neural Network Outputs:", outputs)
}
