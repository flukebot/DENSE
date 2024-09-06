package dense

import (
	"fmt"
	"runtime"
	"sync"
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

// Single-threaded benchmark for float32 operations
func benchmarkFloat32SingleThreaded(duration time.Duration) int {
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < duration {
		ops += floatingPointOperations32(1000)
	}
	return ops
}

// Single-threaded benchmark for float64 operations
func benchmarkFloat64SingleThreaded(duration time.Duration) int {
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < duration {
		ops += floatingPointOperations64(1000)
	}
	return ops
}

// Multi-threaded worker for float32 operations
func benchmarkFloat32Worker(duration time.Duration, wg *sync.WaitGroup, opsChan chan int) {
	defer wg.Done()
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < duration {
		ops += floatingPointOperations32(1000)
	}
	opsChan <- ops
}

// Multi-threaded worker for float64 operations
func benchmarkFloat64Worker(duration time.Duration, wg *sync.WaitGroup, opsChan chan int) {
	defer wg.Done()
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < duration {
		ops += floatingPointOperations64(1000)
	}
	opsChan <- ops
}

// Multi-threaded benchmark for float32 operations
func benchmarkFloat32MultiThreaded(duration time.Duration) int {
	numCores := runtime.NumCPU()
	var wg sync.WaitGroup
	opsChan := make(chan int, numCores)

	for i := 0; i < numCores; i++ {
		wg.Add(1)
		go benchmarkFloat32Worker(duration, &wg, opsChan)
	}

	wg.Wait()
	close(opsChan)

	totalOps := 0
	for ops := range opsChan {
		totalOps += ops
	}
	return totalOps
}

// Multi-threaded benchmark for float64 operations
func benchmarkFloat64MultiThreaded(duration time.Duration) int {
	numCores := runtime.NumCPU()
	var wg sync.WaitGroup
	opsChan := make(chan int, numCores)

	for i := 0; i < numCores; i++ {
		wg.Add(1)
		go benchmarkFloat64Worker(duration, &wg, opsChan)
	}

	wg.Wait()
	close(opsChan)

	totalOps := 0
	for ops := range opsChan {
		totalOps += ops
	}
	return totalOps
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

func RunBenchmarks() (string, string, string, string) {
	duration := 5 * time.Second // Run for 5 seconds to smooth out spikes

	// Single-threaded float32
	fmt.Println("Benchmarking single-threaded float32 operations for 5 seconds...")
	ops32Single := benchmarkFloat32SingleThreaded(duration) / 5
	formattedOps32Single := FormatNumber(ops32Single)
	fmt.Printf("Single-threaded Float32 operations per second: %s\n", formattedOps32Single)

	// Single-threaded float64
	fmt.Println("Benchmarking single-threaded float64 operations for 5 seconds...")
	ops64Single := benchmarkFloat64SingleThreaded(duration) / 5
	formattedOps64Single := FormatNumber(ops64Single)
	fmt.Printf("Single-threaded Float64 operations per second: %s\n", formattedOps64Single)

	// Multi-threaded float32
	fmt.Println("Benchmarking multi-threaded float32 operations for 5 seconds...")
	ops32Multi := benchmarkFloat32MultiThreaded(duration) / 5
	formattedOps32Multi := FormatNumber(ops32Multi)
	fmt.Printf("Multi-threaded Float32 operations per second: %s\n", formattedOps32Multi)

	// Multi-threaded float64
	fmt.Println("Benchmarking multi-threaded float64 operations for 5 seconds...")
	ops64Multi := benchmarkFloat64MultiThreaded(duration) / 5
	formattedOps64Multi := FormatNumber(ops64Multi)
	fmt.Printf("Multi-threaded Float64 operations per second: %s\n", formattedOps64Multi)

	return formattedOps32Single, formattedOps64Single, formattedOps32Multi, formattedOps64Multi
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

func RunAllBenchmarks() (string, string, string, string, string, string, string, string) {
	// Run floating-point operation benchmarks
	ops32Single, ops64Single, ops32Multi, ops64Multi := RunBenchmarks()

	// Estimate maximum layers and nodes for single-threaded
	maxLayers32Single, maxLayers64Single := EstimateMaxLayersAndNodes(
		int(benchmarkFloat32SingleThreaded(5*time.Second))/5,
		int(benchmarkFloat64SingleThreaded(5*time.Second))/5,
	)

	// Estimate maximum layers and nodes for multi-threaded
	maxLayers32Multi, maxLayers64Multi := EstimateMaxLayersAndNodes(
		int(benchmarkFloat32MultiThreaded(5*time.Second))/5,
		int(benchmarkFloat64MultiThreaded(5*time.Second))/5,
	)

	// Print the estimates
	fmt.Printf("Estimated maximum layers with single-threaded float32: %s (with 1000 nodes per layer)\n", maxLayers32Single)
	fmt.Printf("Estimated maximum layers with single-threaded float64: %s (with 1000 nodes per layer)\n", maxLayers64Single)
	fmt.Printf("Estimated maximum layers with multi-threaded float32: %s (with 1000 nodes per layer)\n", maxLayers32Multi)
	fmt.Printf("Estimated maximum layers with multi-threaded float64: %s (with 1000 nodes per layer)\n", maxLayers64Multi)

	// Return both the floating-point operation results and the neural network estimates
	return ops32Single, ops64Single, ops32Multi, ops64Multi, maxLayers32Single, maxLayers64Single, maxLayers32Multi, maxLayers64Multi
}
