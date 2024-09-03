package cortexbuilder

import (
	"fmt"
	"time"
)

func floatingPointOperations32(count int) int {
	var a, b float32 = 1.1, 2.2
	var ops int

	for i := 0; i < count; i++ {
		a = a * b
		b = b + a
		ops++
	}

	return ops
}

func floatingPointOperations64(count int) int {
	var a, b float64 = 1.1, 2.2
	var ops int

	for i := 0; i < count; i++ {
		a = a * b
		b = b + a
		ops++
	}

	return ops
}

func benchmarkFloat32(duration time.Duration) int {
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < duration {
		ops += floatingPointOperations32(1000)
	}
	return ops
}

func benchmarkFloat64(duration time.Duration) int {
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < duration {
		ops += floatingPointOperations64(1000)
	}
	return ops
}

// FormatNumber formats large numbers into a more readable string with suffixes like Thousand, Million, Billion, etc.
func FormatNumber(num int) string {
	switch {
	case num >= 1e12:
		return fmt.Sprintf("%.2f Trillion", float64(num)/1e12)
	case num >= 1e9:
		return fmt.Sprintf("%.2f Billion", float64(num)/1e9)
	case num >= 1e6:
		return fmt.Sprintf("%.2f Million", float64(num)/1e6)
	case num >= 1e3:
		return fmt.Sprintf("%.2f Thousand", float64(num)/1e3)
	default:
		return fmt.Sprintf("%d", num)
	}
}

func RunBenchmarks() (string, string) {
	duration := 1 * time.Second

	fmt.Println("Benchmarking float32 operations for 1 second...")
	ops32 := benchmarkFloat32(duration)
	formattedOps32 := FormatNumber(ops32)
	fmt.Printf("Float32 operations per second: %s\n", formattedOps32)

	fmt.Println("Benchmarking float64 operations for 1 second...")
	ops64 := benchmarkFloat64(duration)
	formattedOps64 := FormatNumber(ops64)
	fmt.Printf("Float64 operations per second: %s\n", formattedOps64)

	return formattedOps32, formattedOps64
}

func EstimateMaxLayersAndNodes(ops32, ops64 int) (string, string) {
	// Assuming a simple fully connected network where each layer is fully connected to the previous one
	// and each operation is a multiply-add (MAC), we'll estimate the maximum number of layers and nodes.

	// For simplicity, we'll assume that each layer has a constant number of nodes.
	const nodesPerLayer = 1000

	// Estimate maximum layers possible for float32 operations
	maxLayers32 := ops32 / (nodesPerLayer * nodesPerLayer)
	// Estimate maximum layers possible for float64 operations
	maxLayers64 := ops64 / (nodesPerLayer * nodesPerLayer)

	return FormatNumber(maxLayers32), FormatNumber(maxLayers64)
}

func RunAllBenchmarks() (string, string, string, string) {
	// Run floating-point operation benchmarks
	ops32, ops64 := RunBenchmarks()

	// Estimate maximum layers and nodes
	maxLayers32, maxLayers64 := EstimateMaxLayersAndNodes(
		int(benchmarkFloat32(1*time.Second)), 
		int(benchmarkFloat64(1*time.Second)),
	)

	// Print the estimates
	fmt.Printf("Estimated maximum layers with float32: %s (with 1000 nodes per layer)\n", maxLayers32)
	fmt.Printf("Estimated maximum layers with float64: %s (with 1000 nodes per layer)\n", maxLayers64)

	// Return both the floating-point operation results and the neural network estimates
	return ops32, ops64, maxLayers32, maxLayers64
}
