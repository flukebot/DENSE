package main

import (
	"encoding/json"
	"syscall/js"

	"dense" // Import the dense package directly
)

func feedforwardWrapper(this js.Value, p []js.Value) interface{} {
	var config dense.NetworkConfig
	inputs := make(map[string]float64)

	err := json.Unmarshal([]byte(p[0].String()), &config)
	if err != nil {
		return err.Error()
	}
	err = json.Unmarshal([]byte(p[1].String()), &inputs)
	if err != nil {
		return err.Error()
	}

	outputs := dense.Feedforward(&config, inputs)

	outputJSON, err := json.Marshal(outputs)
	if err != nil {
		return err.Error()
	}

	return string(outputJSON)
}

func main() {
	js.Global().Set("feedforward", js.FuncOf(feedforwardWrapper))
	c := make(chan struct{}, 0)
	<-c
}
