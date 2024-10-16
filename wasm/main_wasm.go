package main

import (
	"encoding/json"
	"syscall/js"

	"dense" // Import the dense package directly
)

func feedforwardWrapper(this js.Value, p []js.Value) interface{} {
	var config dense.NetworkConfig
	inputs := make(map[string]interface{}) // Use interface{} to allow flexibility with input types

	// Parse JSON input for config
	err := json.Unmarshal([]byte(p[0].String()), &config)
	if err != nil {
		return js.ValueOf(err.Error())
	}

	// Parse JSON input for the inputs
	err = json.Unmarshal([]byte(p[1].String()), &inputs)
	if err != nil {
		return js.ValueOf(err.Error())
	}

	// Call the Feedforward function
	outputs := dense.Feedforward(&config, inputs)

	// Convert outputs to JSON for return
	outputJSON, err := json.Marshal(outputs)
	if err != nil {
		return js.ValueOf(err.Error())
	}

	return js.ValueOf(string(outputJSON))
}

func main() {
	// Export the feedforward function to JavaScript
	js.Global().Set("feedforward", js.FuncOf(feedforwardWrapper))

	// Prevent the Go program from exiting
	c := make(chan struct{}, 0)
	<-c
}
