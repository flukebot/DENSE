package main

import (
	"dense"
	"encoding/json"
	"fmt"
	"image/color"
	"image/jpeg"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"strings"
	"time"
	"strconv"
)

// MNISTImageData represents the structure of each entry in mnistData.json
type MNISTImageData struct {
	FileName string `json:"file_name"`
	Label    int    `json:"label"`
}

// TopModel represents a model and its accuracy
type TopModel struct {
	Config   *dense.NetworkConfig
	Accuracy float64
	Path     string
}

var jsonFilePath string
var mnistData []MNISTImageData


// TestModelPerformance compares the performance of full model evaluation vs. saved layer state.
func TestModelPerformance(modelConfig *dense.NetworkConfig, testData []MNISTImageData, modelFilePath string) {
    // Get the index of the last hidden layer
    layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

    fmt.Println("Starting full model evaluation...")
    // Timing full model evaluation
    startFullEval := time.Now()
    fullEvalOutputs := make([]map[string]float64, len(testData))
    
    for i, data := range testData {
        inputs := convertImageToInputs(data.FileName)
        fullEvalOutputs[i] = dense.Feedforward(modelConfig, inputs)
    }
    durationFullEval := time.Since(startFullEval)
    fmt.Printf("Full model evaluation took: %s\n", durationFullEval)

    fmt.Println("Loading saved layer state...")
    // No need to load the saved layer state here

    fmt.Println("Starting evaluation from saved layer state...")
    // Timing evaluation from the saved layer state
    startSavedEval := time.Now()
    savedEvalOutputs := make([]map[string]float64, len(testData))
    for i := range testData {
        inputID := fmt.Sprintf("%d", i)
        savedLayerData := loadSavedLayerState(modelFilePath, layerStateNumber, inputID)
        savedEvalOutputs[i] = runFromSavedLayer(modelConfig, savedLayerData, layerStateNumber)
    }
    durationSavedEval := time.Since(startSavedEval)
    fmt.Printf("Evaluation from saved layer state took: %s\n", durationSavedEval)

    // Compare outputs
    consistent := true
    for i := range testData {
        fmt.Println(fullEvalOutputs[i], savedEvalOutputs[i])
        if !compareOutputs(fullEvalOutputs[i], savedEvalOutputs[i]) {
            fmt.Printf("Outputs differ at index %d!\n", i)
            consistent = false
            break
        }
    }

    if consistent {
        fmt.Println("All outputs are consistent!")
    }

	
}



// runFullModelEvaluation runs the full model evaluation and returns the outputs.
func runFullModelEvaluation(modelConfig *dense.NetworkConfig, testData []MNISTImageData) map[string]float64 {
    fmt.Println("Running full model evaluation...")
    correct := 0
    for i, data := range testData {
        inputs := convertImageToInputs(data.FileName)
        outputPredicted := dense.Feedforward(modelConfig, inputs)
        predictedLabel := getMaxIndex(outputPredicted)
        if predictedLabel == data.Label {
            correct++
        }
        if i%100 == 0 {
            fmt.Printf("Processed %d/%d samples\n", i+1, len(testData))
        }
    }
    accuracy := float64(correct) / float64(len(testData))
    fmt.Printf("Full model accuracy: %.2f%%\n", accuracy*100)
    return dense.Feedforward(modelConfig, convertImageToInputs(testData[0].FileName)) // Returning outputs for comparison
}


// loadSavedLayerState loads the saved layer state from the CSV file for a specific inputID.
func loadSavedLayerState(modelFilePath string, layerStateNumber int, inputID string) interface{} {
    dir, file := filepath.Split(modelFilePath)
    modelName := strings.TrimSuffix(file, filepath.Ext(file))
    layerCSVFilePath := filepath.Join(dir, modelName, fmt.Sprintf("layer_%d.csv", layerStateNumber))

    // Load CSV data here (this function should return the saved layer state as interface{} for further processing)
    // You can adapt this function based on your CSV reading logic
    savedLayerData := dense.LoadCSVLayerState(layerCSVFilePath, inputID)

    return savedLayerData
}


// runFromSavedLayer runs the model evaluation starting from a saved layer state.
func runFromSavedLayer(modelConfig *dense.NetworkConfig, savedLayerData interface{}, startLayer int) map[string]float64 {
    return dense.ContinueFeedforward(modelConfig, savedLayerData, startLayer)
}

// compareOutputs compares the outputs of the full model evaluation and the saved layer state evaluation.
func compareOutputs(output1, output2 map[string]float64) bool {
    if len(output1) != len(output2) {
        return false
    }
    for key, value1 := range output1 {
        value2, exists := output2[key]
        if !exists || math.Abs(value1-value2) > 1e-6 {
            return false
        }
    }
    return true
}

func main() {
	fmt.Println("Starting CNN train and host")
	jsonFilePath = "./host/mnistData.json"
	// Check if the MNIST directory exists, and run setup if it doesn't
	mnistDir := "./host/MNIST"
	if !dense.CheckDirExists(mnistDir) {
		fmt.Println("MNIST directory doesn't exist, running setupMNIST()")
		setupMNIST()
	} else {
		fmt.Println("MNIST directory already exists, skipping setup.")
	}


	
	LoadMNISTData()
	

	// Set up the model configuration
	projectName := "AIModelTestProject"
	inputSize := 28 * 28               // Input size for MNIST data
	outputSize := 10                   // Output size for MNIST digits (0-9)
	outputTypes := []string{"softmax"} // Activation type for output layer
	//mnistDataFilePath := "./host/mnistData.json"
	//percentageTrain := 0.8
	numModels := 10
	//generationNum := 500
	/*modelConfig := dense.CreateRandomNetworkConfig(inputSize, outputSize, outputTypes, "id1", projectName)


	

	// Define the path to the MNIST data JSON file
	

	// Train and evaluate the model using 80% of the data for training
	accuracy, err := EvaluateModel(jsonFilePath, modelConfig, 0.8)
	if err != nil {
		log.Fatalf("Failed to train and evaluate model: %v", err)
	}

	// Display the model accuracy
	fmt.Printf("Model accuracy: %.2f%%\n", accuracy*100)*/

	// Check if the generation folder exists, and generate models if it doesn't
	generationDir := "./host/generations/0"
	if !dense.CheckDirExists(generationDir) {
		fmt.Println("Generation folder doesn't exist, generating models.")
		// Number of models to generate

		// Generate the models and save them to host/generations/0
		if err := GenerateModels(numModels, inputSize, outputSize, outputTypes, projectName); err != nil {
			log.Fatalf("Failed to generate models: %v", err)
		}
		fmt.Println("Model generation complete.")

	} else {
		fmt.Println("Generation folder already exists, skipping model generation.")
	}

	saveLayerStates(generationDir)

	GenCycleLocalTesting(generationDir)

	//fmt.Println(mnistData[10007])

	// Load model and test data here
    
	 // Save sharded layer states for testing
	 //fmt.Println("Saving sharded layer states...")
	 //SaveShardedLayerStates(modelConfig, mnistData[:10], "./host/generations/0/model_0.json")
 
	 // Test performance of the model using sharded layer states



    //TestModelPerformance(modelConfig, mnistData, "./host/generations/0/model_0.json")
	//CompareModelOutputsWithLoadTimesSingleLoad(modelConfig, mnistData[:10], "./host/generations/0/model_0.json")
}

func testPer(){
	modelConfig, err := loadModel("./host/generations/0/model_0.json")
    if err != nil {
        fmt.Println("Failed to load model:", err)
        return
    }

	 fmt.Println("Testing performance with sharded layer states...")
	 TestShardedModelPerformanceMultithreaded(modelConfig, mnistData[:40000], "./host/generations/0/model_0.json")
}

//once built will cycle through 500 gens then build static mnist distributed version poc
func GenCycleLocalTesting(generationDir string) {
    files, err := ioutil.ReadDir(generationDir)
    if err != nil {
        fmt.Printf("Failed to read models directory: %v\n", err)
        return
    }

    totalFiles := len(files)
    batchSize := 2 // Number of models to process concurrently

    var wg sync.WaitGroup
    semaphore := make(chan struct{}, batchSize) // Semaphore to limit concurrency

    for batchIndex := 0; batchIndex < totalFiles; batchIndex += batchSize {
        endIndex := batchIndex + batchSize
        if endIndex > totalFiles {
            endIndex = totalFiles
        }

        for i := batchIndex; i < endIndex; i++ {

			if filepath.Ext(files[i].Name()) != ".json" {
                continue // Skip non-JSON files
            }

            modelFilePath := filepath.Join(generationDir, files[i].Name())
			modelFilePathFolder := strings.TrimSuffix(modelFilePath, filepath.Ext(modelFilePath))

            wg.Add(1)
            semaphore <- struct{}{} // Acquire a slot in the semaphore

            go func(modelFilePath string, modelFilePathFolder string) {
                defer wg.Done()
                defer func() { <-semaphore }() // Release the semaphore when done

                //fmt.Printf("Processing model: %s\n", modelFilePath)
                //processModelEvalState(modelFilePath)

				

				highestFolder, err := dense.FindHighestNumberedFolder(modelFilePathFolder, "layer", "learnedornot")
				if err != nil {
					log.Fatal(err)
				}
				fmt.Println("Highest numbered folder:", highestFolder)
				layerNum,_ := dense.ExtractDigitsToInt(highestFolder)
				fmt.Println(layerNum)

				layerOfNotLearned := filepath.Join(modelFilePathFolder, highestFolder)
				lstEvalsTryingToLearn,_ := dense.GetFilesWithExtension(layerOfNotLearned,".false",10,false)
				fmt.Println(lstEvalsTryingToLearn)

				//mutations to try against shards
				//dense.AppendNewLayerFullConnections(config *NetworkConfig, numNewNeurons int)
				//dense.AppendMultipleLayers(config *NetworkConfig, numNewLayers int, numNewNeurons int)

				for _, dataShard := range lstEvalsTryingToLearn {
					//fmt.Println(indexShards,dataShard)
					//fmt.Println(modelFilePathFolder + "/layer_" + strconv.Itoa(layerNum) + "_shards/" + dataShard + ".csv")

					inputIDNumber,_ := dense.ExtractDigitsToInt(dataShard)
					savedLayerData := dense.LoadShardedLayerState(modelFilePath, layerNum, strconv.Itoa(inputIDNumber))
                	

					modelConfig, err := loadModel(modelFilePathFolder + ".json")
					if err != nil {
						fmt.Println("Failed to load model:", err)
					}else{
						result := dense.ContinueFeedforward(modelConfig  , savedLayerData, layerNum)
						//fmt.Println(savedLayerData)
						//fmt.Println(result)

						predictedLabel := getMaxIndex(result)

						// Compare with the actual label
						if predictedLabel == mnistData[inputIDNumber].Label {
                            fmt.Println("MATCH!!!")
						}else{
                            //fmt.Println(predictedLabel)
                            //fmt.Println(mnistData[inputIDNumber].Label)

                            ApplyMutations(modelFilePathFolder, inputIDNumber, layerNum, savedLayerData)

                        }

						//mnistData[10007])
					}

					
				}

            }(modelFilePath,modelFilePathFolder)
        }

        // Wait for all goroutines in this batch to complete before moving to the next batch
        wg.Wait()
    }

    fmt.Println("All models processed.")
}

func ApplyMutations(modelFilePathFolder string, inputIDNumber int, layerNum int, savedLayerData interface{}) {
    var wg sync.WaitGroup    // WaitGroup to manage goroutines
    var mu sync.Mutex        // Mutex to protect shared resources
    mutationAttempts := 100   // Number of mutation attempts
    var foundMatch bool      // Flag to track if we found a matching prediction

    // Channel to capture the result
    mutationResults := make(chan bool, mutationAttempts)

    for i := 0; i < mutationAttempts; i++ {
        wg.Add(1) // Increment WaitGroup counter

        go func(iteration int) {
            defer wg.Done() // Decrement WaitGroup counter when done

            // Load the model configuration inside each goroutine
            modelConfig, err := loadModel(modelFilePathFolder + ".json")
            if err != nil {
                fmt.Println("Failed to load model:", err)
                return
            }

            // Randomize the number of neurons or filter size (for CNN layers) between 10 and 128
            numNewNeuronsOrFilters := rand.Intn(119) + 10

            // Randomly choose between dense (FFNN), CNN, and LSTM mutation
            mutationType := rand.Intn(6) // 6 mutation types

            // Apply the mutation based on type
            switch mutationType {
            case 0:
                dense.AppendNewLayerFullConnections(modelConfig, numNewNeuronsOrFilters)
            case 1:
                numNewLayers := rand.Intn(3) + 1
                dense.AppendMultipleLayers(modelConfig, numNewLayers, numNewNeuronsOrFilters)
            case 2:
                dense.AddCNNLayerAtRandomPosition(modelConfig, numNewNeuronsOrFilters)
            case 3:
                numNewLayers := rand.Intn(3) + 1
                dense.AddMultipleCNNLayers(modelConfig, 100, numNewLayers)
            case 4:
                dense.AddLSTMLayerAtRandomPosition(modelConfig, 100)
            case 5:
                numNewLayers := rand.Intn(3) + 1
                dense.AddMultipleLayers(modelConfig, numNewLayers)
            }

            // *** Reattach the output layer after applying mutations ***
            numOutputs := 10 // Number of output neurons (for example, for classification of MNIST digits 0-9)
            outputActivationTypes := []string{"softmax"} // Activation type for the output layer
            dense.ReattachOutputLayer(modelConfig, numOutputs, outputActivationTypes)

            // *** Continue the feedforward process from the saved layer state ***
            result := dense.ContinueFeedforward(modelConfig, savedLayerData, layerNum)
            mutatedPredictedLabel := getMaxIndex(result)
            
            // If the prediction matches the actual label, mark the match
            if mutatedPredictedLabel == mnistData[inputIDNumber].Label {
                mu.Lock()
                if !foundMatch { // Check and set foundMatch in a thread-safe manner
                    foundMatch = true
                    //fmt.Printf("Match found on iteration %d\n", iteration)
                    mutatedAccuracy := EvaluateModelAccuracy(modelConfig, mnistData)
                    baselineAccuracy := modelConfig.Metadata.LastTestAccuracy
                    fmt.Printf("Old Accuracy: %.2f%%\n", baselineAccuracy*100)
                    fmt.Printf("New Accuracy: %.2f%%\n", mutatedAccuracy*100)
                    
                }
                mu.Unlock()
                mutationResults <- true
            } else {
                mutationResults <- false
            }

        }(i)
    }

    // Wait for all goroutines to finish
    wg.Wait()
    close(mutationResults)

    // Check if any of the mutations resulted in a match
    for result := range mutationResults {
        if result {
            fmt.Println("Successful mutation found!")
           
            break
        }
    }
}


// EvaluateModelAccuracy evaluates the model's accuracy on the entire test dataset.
func EvaluateModelAccuracy(modelConfig *dense.NetworkConfig, testData []MNISTImageData) float64 {
    var correct int
    total := len(testData)
    var mu sync.Mutex
    var wg sync.WaitGroup

    numWorkers := 10
    batchSize := total / numWorkers
    if total%numWorkers != 0 {
        batchSize++
    }

    for i := 0; i < numWorkers; i++ {
        start := i * batchSize
        end := start + batchSize
        if end > total {
            end = total
        }
        if start >= end {
            break
        }

        wg.Add(1)
        go func(start, end int) {
            defer wg.Done()
            localCorrect := 0
            for j := start; j < end; j++ {
                data := testData[j]
                inputs := convertImageToInputs(data.FileName)
                output := dense.Feedforward(modelConfig, inputs)
                predictedLabel := getMaxIndex(output)
                if predictedLabel == data.Label {
                    localCorrect++
                }
            }
            mu.Lock()
            correct += localCorrect
            mu.Unlock()
        }(start, end)
    }

    wg.Wait()
    accuracy := float64(correct) / float64(total)
    fmt.Printf("Model accuracy: %.2f%%\n", accuracy*100)
    return accuracy
}







// TestShardedModelPerformance compares the performance of full model evaluation vs. sharded layer state.
func TestShardedModelPerformance(modelConfig *dense.NetworkConfig, testData []MNISTImageData, modelFilePath string) {
    layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

    fmt.Println("Starting full model evaluation...")
    startFullEval := time.Now()
    fullEvalOutputs := make([]map[string]float64, len(testData))

    for i, data := range testData {
        inputs := convertImageToInputs(data.FileName)
        fullEvalOutputs[i] = dense.Feedforward(modelConfig, inputs)
    }
    durationFullEval := time.Since(startFullEval)
    fmt.Printf("Full model evaluation took: %s\n", durationFullEval)

    fmt.Println("Starting evaluation from sharded layer state...")
    startSavedEval := time.Now()
    savedEvalOutputs := make([]map[string]float64, len(testData))
    for i := range testData {
        inputID := fmt.Sprintf("%d", i)
        savedLayerData := dense.LoadShardedLayerState(modelFilePath, layerStateNumber, inputID)
        savedEvalOutputs[i] = dense.ContinueFeedforward(modelConfig, savedLayerData, layerStateNumber)
    }
    durationSavedEval := time.Since(startSavedEval)
    fmt.Printf("Evaluation from sharded layer state took: %s\n", durationSavedEval)

    consistent := true
    for i := range testData {
        if !compareOutputs(fullEvalOutputs[i], savedEvalOutputs[i]) {
            fmt.Printf("Outputs differ at index %d!\n", i)
            consistent = false
            break
        }
    }

    if consistent {
        fmt.Println("All outputs are consistent!")
    }
}

func TestShardedModelPerformanceMultithreaded(modelConfig *dense.NetworkConfig, testData []MNISTImageData, modelFilePath string) {
    layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

    // Prepare synchronization tools for multithreading
    var wg sync.WaitGroup
    var mu sync.Mutex // Mutex to protect shared resources

    fullEvalOutputs := make([]map[string]float64, len(testData))
    savedEvalOutputs := make([]map[string]float64, len(testData))

    fmt.Println("Starting full model evaluation...")

    // Start timing for full evaluation
    startFullEval := time.Now()

    // Multithreaded Full Model Evaluation
    numWorkers := 10
    batchSize := len(testData) / numWorkers
    if len(testData)%numWorkers != 0 {
        batchSize++
    }

    // Launch goroutines for full evaluation
    for i := 0; i < numWorkers; i++ {
        start := i * batchSize
        end := start + batchSize
        if end > len(testData) {
            end = len(testData)
        }
        if start >= end {
            break
        }

        wg.Add(1)
        go func(start, end int) {
            defer wg.Done()
            for j := start; j < end; j++ {
                inputs := convertImageToInputs(testData[j].FileName)
                result := dense.Feedforward(modelConfig, inputs)

                // Use mutex to ensure thread-safe access to shared resources
                mu.Lock()
                fullEvalOutputs[j] = result
                mu.Unlock()
            }
        }(start, end)
    }

    // Wait for full model evaluation to finish
    wg.Wait()

    durationFullEval := time.Since(startFullEval)
    fmt.Printf("Full model evaluation took: %s\n", durationFullEval)

    fmt.Println("Starting evaluation from sharded layer state...")

    // Start timing for saved layer state evaluation
    startSavedEval := time.Now()

    // Multithreaded Saved Layer State Evaluation
    for i := 0; i < numWorkers; i++ {
        start := i * batchSize
        end := start + batchSize
        if end > len(testData) {
            end = len(testData)
        }
        if start >= end {
            break
        }

        wg.Add(1)
        go func(start, end int) {
            defer wg.Done()
            for j := start; j < end; j++ {
                inputID := fmt.Sprintf("%d", j)
                savedLayerData := dense.LoadShardedLayerState(modelFilePath, layerStateNumber, inputID)
                result := dense.ContinueFeedforward(modelConfig, savedLayerData, layerStateNumber)

                // Use mutex to ensure thread-safe access to shared resources
                mu.Lock()
                savedEvalOutputs[j] = result
                mu.Unlock()
            }
        }(start, end)
    }

    // Wait for saved layer state evaluation to finish
    wg.Wait()

    durationSavedEval := time.Since(startSavedEval)
    fmt.Printf("Evaluation from sharded layer state took: %s\n", durationSavedEval)

    // Compare the outputs to ensure consistency
    consistent := true
    for i := range testData {
        if !compareOutputs(fullEvalOutputs[i], savedEvalOutputs[i]) {
            fmt.Printf("Outputs differ at index %d!\n", i)
            consistent = false
            break
        }
    }

    if consistent {
        fmt.Println("All outputs are consistent!")
    }
}






func CompareModelOutputsWithLoadTimesSingleLoad(modelConfig *dense.NetworkConfig, testData []MNISTImageData, modelFilePath string) {
    // Get the index of the last hidden layer
    layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

    fmt.Println("Comparing outputs for the top", len(testData), "images.")

    fullModelDurations := make([]time.Duration, len(testData))
    savedStateDurations := make([]time.Duration, len(testData))
    loadStateDuration := time.Duration(0) // To track total loading time
    outputsMatch := true

    // Load saved layer states once (for all inputs)
    savedLayerData := make(map[string]interface{}, len(testData))
    startLoad := time.Now()
    for i := range testData {
        inputID := fmt.Sprintf("%d", i)
        savedLayerData[inputID] = loadSavedLayerState(modelFilePath, layerStateNumber, inputID)
    }
    loadStateDuration = time.Since(startLoad)

    for i, data := range testData {
        inputs := convertImageToInputs(data.FileName)
        inputID := fmt.Sprintf("%d", i)

        // Run full model
        startFull := time.Now()
        fullOutput := dense.Feedforward(modelConfig, inputs)
        fullModelDurations[i] = time.Since(startFull)

        // Run model starting from saved layer state
        startSaved := time.Now()
        state := savedLayerData[inputID]
        if state == nil {
            fmt.Printf("Saved layer data not found for input %s. Skipping.\n", inputID)
            continue
        }
        savedOutput := dense.ContinueFeedforward(modelConfig, state, layerStateNumber)
        savedStateDurations[i] = time.Since(startSaved) // Only timing feedforward, not loading

        fmt.Println(fullOutput)
        fmt.Println("----------")
        fmt.Println(savedOutput)
        // Compare outputs
        if !compareOutputs(fullOutput, savedOutput) {
            fmt.Printf("Outputs differ at index %d!\n", i)
            outputsMatch = false
        } else {
            fmt.Printf("Outputs match for index %d.\n", i)
        }
    }

    // Calculate total durations
    var totalFullDuration, totalSavedDuration time.Duration
    for i := range testData {
        totalFullDuration += fullModelDurations[i]
        totalSavedDuration += savedStateDurations[i]
    }

    fmt.Printf("Total time for full model evaluation: %s\n", totalFullDuration)
    fmt.Printf("Total time for evaluation from saved layer state (excluding load): %s\n", totalSavedDuration)
    fmt.Printf("Total time spent loading layer states: %s\n", loadStateDuration)

    if outputsMatch {
        fmt.Println("All outputs match!")
    } else {
        fmt.Println("There were mismatches in the outputs.")
    }
}


func CompareModelOutputsWithLoadTimes(modelConfig *dense.NetworkConfig, testData []MNISTImageData, modelFilePath string) {
	// Get the index of the last hidden layer
	layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

	fmt.Println("Comparing outputs for the top", len(testData), "images.")

	fullModelDurations := make([]time.Duration, len(testData))
	savedStateDurations := make([]time.Duration, len(testData))
	outputsMatch := true

	for i, data := range testData {
		inputs := convertImageToInputs(data.FileName)
		inputID := fmt.Sprintf("%d", i)

		// Run full model
		startFull := time.Now()
		fullOutput := dense.Feedforward(modelConfig, inputs)
		fullModelDurations[i] = time.Since(startFull)

		// Run model starting from saved layer state
		// Include the time to load the saved layer state
		startSaved := time.Now()

		// Load saved layer state for this input
		savedLayerData := loadSavedLayerState(modelFilePath, layerStateNumber, inputID)
		if savedLayerData == nil {
			fmt.Printf("Saved layer data not found for input %s. Skipping.\n", inputID)
			continue
		}

		savedOutput := dense.ContinueFeedforward(modelConfig, savedLayerData, layerStateNumber)
		savedStateDurations[i] = time.Since(startSaved) // This now includes the state loading time

		fmt.Println(fullOutput)
		fmt.Println("----------")
		fmt.Println(savedOutput)
		// Compare outputs
		if !compareOutputs(fullOutput, savedOutput) {
			fmt.Printf("Outputs differ at index %d!\n", i)
			outputsMatch = false
		} else {
			fmt.Printf("Outputs match for index %d.\n", i)
		}
	}

	// Calculate total durations
	var totalFullDuration, totalSavedDuration time.Duration
	for i := range testData {
		totalFullDuration += fullModelDurations[i]
		totalSavedDuration += savedStateDurations[i]
	}

	fmt.Printf("Total time for full model evaluation: %s\n", totalFullDuration)
	fmt.Printf("Total time for evaluation from saved layer state (including load): %s\n", totalSavedDuration)

	if outputsMatch {
		fmt.Println("All outputs match!")
	} else {
		fmt.Println("There were mismatches in the outputs.")
	}
}


// CompareModelOutputs compares the outputs of the full model and the model starting from the saved layer state
// for the top N images in the MNIST dataset.
func CompareModelOutputs(modelConfig *dense.NetworkConfig, testData []MNISTImageData, modelFilePath string) {
	// Get the index of the last hidden layer
	layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

	fmt.Println("Comparing outputs for the top", len(testData), "images.")

	fullModelDurations := make([]time.Duration, len(testData))
	savedStateDurations := make([]time.Duration, len(testData))
	outputsMatch := true

	for i, data := range testData {
		inputs := convertImageToInputs(data.FileName)
		inputID := fmt.Sprintf("%d", i)

		// Run full model
		startFull := time.Now()
		fullOutput := dense.Feedforward(modelConfig, inputs)
		fullModelDurations[i] = time.Since(startFull)

		// Run model starting from saved layer state
		// Load saved layer state for this input
		savedLayerData := loadSavedLayerState(modelFilePath, layerStateNumber, inputID)
		if savedLayerData == nil {
			fmt.Printf("Saved layer data not found for input %s. Skipping.\n", inputID)
			continue
		}

		startSaved := time.Now()
		savedOutput := dense.ContinueFeedforward(modelConfig, savedLayerData, layerStateNumber)
		savedStateDurations[i] = time.Since(startSaved)
		fmt.Println(fullOutput)
		fmt.Println("----------")
		fmt.Println(savedOutput)
		// Compare outputs
		if !compareOutputs(fullOutput, savedOutput) {
			fmt.Printf("Outputs differ at index %d!\n", i)
			outputsMatch = false
		} else {
			fmt.Printf("Outputs match for index %d.\n", i)
		}
	}

	// Calculate total durations
	var totalFullDuration, totalSavedDuration time.Duration
	for i := range testData {
		totalFullDuration += fullModelDurations[i]
		totalSavedDuration += savedStateDurations[i]
	}

	fmt.Printf("Total time for full model evaluation: %s\n", totalFullDuration)
	fmt.Printf("Total time for evaluation from saved layer state: %s\n", totalSavedDuration)

	if outputsMatch {
		fmt.Println("All outputs match!")
	} else {
		fmt.Println("There were mismatches in the outputs.")
	}
}


func saveLayerStates(generationDir string) {
    files, err := ioutil.ReadDir(generationDir)
    if err != nil {
        fmt.Printf("Failed to read models directory: %v\n", err)
        return
    }

    totalFiles := len(files)
    batchSize := 2 // Number of models to process concurrently

    var wg sync.WaitGroup
    semaphore := make(chan struct{}, batchSize) // Semaphore to limit concurrency

    for batchIndex := 0; batchIndex < totalFiles; batchIndex += batchSize {
        endIndex := batchIndex + batchSize
        if endIndex > totalFiles {
            endIndex = totalFiles
        }

        for i := batchIndex; i < endIndex; i++ {

			if filepath.Ext(files[i].Name()) != ".json" {
                continue // Skip non-JSON files
            }

            modelFilePath := filepath.Join(generationDir, files[i].Name())

            wg.Add(1)
            semaphore <- struct{}{} // Acquire a slot in the semaphore

            go func(modelFilePath string) {
                defer wg.Done()
                defer func() { <-semaphore }() // Release the semaphore when done

                fmt.Printf("Processing model: %s\n", modelFilePath)
                processModelEvalState(modelFilePath)
            }(modelFilePath)
        }

        // Wait for all goroutines in this batch to complete before moving to the next batch
        wg.Wait()
    }

    fmt.Println("All models processed.")
}




func processModelEvalState(modelFilePath string) {
	// Replace this with your actual model processing logic
	
	// Simulate some work with a sleep or processing code here
	modelConfig, err := loadModel(modelFilePath)
	if err != nil {
		fmt.Errorf("failed to load model %s: %w", modelFilePath, err)
	}


	// Check if the model has already been evaluated
	if modelConfig.Metadata.LastTestAccuracy > 0 {
		log.Printf("Model %s has already been evaluated with accuracy: %.2f%%. Skipping mutation and evaluation.", modelFilePath, modelConfig.Metadata.LastTestAccuracy*100)
	}else{
		// Shuffle the data to ensure randomness
		//rand.Shuffle(len(mnistData), func(i, j int) {
		//	mnistData[i], mnistData[j] = mnistData[j], mnistData[i]
		//})
		percentageTrain := 0.8
		// Split the data into training and testing sets
		trainSize := int(percentageTrain * float64(len(mnistData)))
		trainData := mnistData[:trainSize]
		//testData := mnistData[trainSize:]

		// Evaluate the model on the testing data
		//fmt.Println("Evaluating the model...")
		//accuracy := evaluateModel(trainData, modelConfig,modelFilePath)
		accuracy := evaluateModelMultiThreaded(trainData, modelConfig,modelFilePath)
		
		fmt.Println("accuracy",accuracy)

		modelConfig.Metadata.LastTestAccuracy = accuracy

		// Save the mutated model back to the same file
		if err := saveModel(modelFilePath, modelConfig); err != nil {
			fmt.Println("failed to save mutated model %s: %w", modelFilePath, err)
		}
	}
	
	

	
}

//step 1-----------------------------

// LoadMNISTData loads the MNIST data from the JSON file and returns an array of MNISTImageData
func LoadMNISTData(){// ([]MNISTImageData, error) {
	jsonFile, _ := os.Open(jsonFilePath)

	defer jsonFile.Close()

	byteValue, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		//return nil, err
	}

	//var mnistData []MNISTImageData
	err = json.Unmarshal(byteValue, &mnistData)
	if err != nil {
		//return nil, err
	}

	//return mnistData, nil
}

func setupMNIST() {
	// Create the directory for MNIST images
	if err := os.MkdirAll("./host/MNIST", os.ModePerm); err != nil {
		log.Fatalf("Failed to create MNIST directory: %v", err)
	}

	// Ensure MNIST data is downloaded and unzipped
	if err := dense.EnsureMNISTDownloads(); err != nil {
		log.Fatalf("Failed to ensure MNIST downloads: %v", err)
	}

	// Load the MNIST data
	mnist, err := dense.LoadMNISTOLD()
	if err != nil {
		log.Fatalf("Failed to load MNIST data: %v", err)
	}

	// Print the number of images and labels for verification
	fmt.Printf("Loaded %d images and %d labels\n", len(mnist.Images), len(mnist.Labels))

	// Save the images and labels to disk
	if err := dense.SaveMNISTImagesAndData(mnist, "./host/MNIST", "./host/mnistData.json"); err != nil {
		log.Fatalf("Failed to save MNIST images and data: %v", err)
	}

	fmt.Println("Successfully saved images and labels.")
}




/*
func evaluateModel(testData []MNISTImageData, modelConfig *dense.NetworkConfig, modelFilePath string) float64 {
    correct := 0

    // Get the index of the last hidden layer
    layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

    // Construct the file path for the layer state CSV file
    dir, file := filepath.Split(modelFilePath)
    modelName := strings.TrimSuffix(file, filepath.Ext(file))
    layerCSVFilePath := filepath.Join(dir, modelName, fmt.Sprintf("layer_%d.csv", layerStateNumber))

    // Check if the layer state CSV file already exists
    if _, err := os.Stat(layerCSVFilePath); err == nil {
        fmt.Printf("Layer state file already exists for model: %s, skipping...\n", layerCSVFilePath)
        return 0.0
    }

    // Proceed with the evaluation
    for idx, data := range testData {
        inputs := convertImageToInputs(data.FileName) // Convert image to input values
        inputID := fmt.Sprintf("%d", idx) // Use index as unique ID
        //outputPredicted := dense.FeedforwardLayerStateSaving(modelConfig, inputs, layerStateNumber, modelFilePath, inputID) // Get predicted outputs
		outputPredicted := dense.FeedforwardLayerStateSavingShard(modelConfig, inputs, layerStateNumber, modelFilePath, inputID)
        // Find the index of the maximum predicted value
        predictedLabel := getMaxIndex(outputPredicted)

        // Compare with the actual label
        if predictedLabel == data.Label {
            correct++
        }
    }

    // Calculate accuracy as the proportion of correct predictions
    accuracy := float64(correct) / float64(len(testData))

    // Ensure the returned accuracy is at least 0.1%
    if accuracy < 0.001 {
        accuracy = 0.001
    }

    return accuracy
}*/

func evaluateModelMultiThreaded(testData []MNISTImageData, modelConfig *dense.NetworkConfig, modelFilePath string) float64 {
    var correct int
    var mu sync.Mutex // To protect the `correct` counter
    var wg sync.WaitGroup

    // Buffer to store all shard data
    var bufferMu sync.Mutex
    shardDataBuffer := make(map[string]interface{})

    // Get the index of the last hidden layer
    layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

	// Create the learnedOrNot folder inside the model directory
	learnedOrNotFolder := dense.CreateLearnedOrNotFolder(modelFilePath, layerStateNumber)


    // Set up the number of goroutines (e.g., 10 workers)
    numWorkers := 10
    batchSize := len(testData) / numWorkers
    if len(testData)%numWorkers != 0 {
        batchSize++
    }

    // Loop through batches of data and assign each to a goroutine
    for i := 0; i < numWorkers; i++ {
        start := i * batchSize
        end := start + batchSize
        if end > len(testData) {
            end = len(testData)
        }
        if start >= end {
            break
        }

        // Increment the wait group counter
        wg.Add(1)

        // Pass the 'start' index to the goroutine
        go func(start int, testDataBatch []MNISTImageData) {
            defer wg.Done() // Decrement the wait group counter when done

            localCorrect := 0
            localShardData := make(map[string]interface{})
			localLearnedStatus := make(map[string]bool)

            for idx, data := range testDataBatch {
                inputs := convertImageToInputs(data.FileName) // Convert image to input values

                // Adjust 'inputID' to be unique across all batches
                inputID := fmt.Sprintf("%d", start+idx)

                // Call the function to get the output and layer state
                outputPredicted, layerState := dense.FeedforwardLayerStateSavingShard(modelConfig, inputs, layerStateNumber, modelFilePath)

                // Save the layer state data into the local buffer
                localShardData[inputID] = layerState

                // Find the index of the maximum predicted value
                predictedLabel := getMaxIndex(outputPredicted)

                // Compare with the actual label
                if predictedLabel == data.Label {
                    localCorrect++
                    localLearnedStatus[inputID] = true  // Prediction was correct
                } else {
                    localLearnedStatus[inputID] = false // Prediction was incorrect
                }
            }

            // Safely update the `correct` counter and add the local shard data to the global buffer
            mu.Lock()
            correct += localCorrect
            mu.Unlock()

            // Store local shard data into the shared buffer
            bufferMu.Lock()
            for key, value := range localShardData {
                shardDataBuffer[key] = value
            }
            bufferMu.Unlock()

			// Save the learned status (true/false) for each inputID
            for inputID, learnedStatus := range localLearnedStatus {
                dense.SaveLearnedOrNot(learnedOrNotFolder, inputID, learnedStatus)
            }

        }(start, testData[start:end])
    }

    // Wait for all goroutines to complete
    wg.Wait()

    // Now perform the actual file writes for all shard data
    for inputID, shardData := range shardDataBuffer {
        dense.SaveShardedLayerState(shardData, modelFilePath, layerStateNumber, inputID)
    }

    // Calculate accuracy as the proportion of correct predictions
    accuracy := float64(correct) / float64(len(testData))

    // Ensure the returned accuracy is at least 0.1%
    if accuracy < 0.001 {
        accuracy = 0.001
    }

    return accuracy
}












// convertImageToInputs loads the image file and converts it into input values for the network
func convertImageToInputs(fileName string) map[string]interface{} {
	// Construct the full file path
	filePath := filepath.Join("./host/MNIST", fileName)

	// Open the image file
	imgFile, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Failed to open image file %s: %v", filePath, err)
	}
	defer imgFile.Close()

	// Decode the JPEG image
	img, err := jpeg.Decode(imgFile)
	if err != nil {
		log.Fatalf("Failed to decode image file %s: %v", filePath, err)
	}

	// Ensure the image is in grayscale format
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	inputs := make(map[string]interface{})
	index := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			colorPixel := img.At(x, y)
			grayColor := color.GrayModel.Convert(colorPixel).(color.Gray)
			pixelValue := float64(grayColor.Y) / 255.0 // Normalize pixel value to [0,1]
			inputs[fmt.Sprintf("input%d", index)] = pixelValue
			index++
		}
	}

	return inputs
}

// convertLabelToOutputs converts the label into a one-hot encoding for output comparison
func convertLabelToOutputs(label int) map[string]float64 {
	outputs := make(map[string]float64)
	for i := 0; i < 10; i++ {
		if i == label {
			outputs[fmt.Sprintf("output%d", i)] = 1.0
		} else {
			outputs[fmt.Sprintf("output%d", i)] = 0.0
		}
	}
	return outputs
}

// getMaxIndex returns the index of the maximum value in the map
func getMaxIndex(outputs map[string]float64) int {
    maxIndex := 0
    maxValue := -math.MaxFloat64
    for i := 0; i < 10; i++ {
        key := fmt.Sprintf("output%d", i)
        value, exists := outputs[key]
        if !exists {
            value = 0.0 // Assume 0 if key doesn't exist
        }
        if value > maxValue {
            maxIndex = i
            maxValue = value
        }
    }
    return maxIndex
}


//---step 3 make bulk of them

// GenerateModels generates a specified number of models and saves them in the host/generations/0 folder.
func GenerateModels(numModels int, inputSize, outputSize int, outputTypes []string, projectName string) error {
	// Create the directory to store the models
	modelDir := "./host/generations/0"
	if err := os.MkdirAll(modelDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}

	// Generate and save models
	for i := 0; i < numModels; i++ {
		modelID := fmt.Sprintf("model_%d", i)
		//modelConfig := dense.CreateRandomNetworkConfig(inputSize, outputSize, outputTypes, modelID, projectName)
		
		//firstLayerNeurons := 2 * inputSize // Double the number of input neurons
		firstLayerNeurons := 128
		modelConfig := dense.CreateCustomNetworkConfig(inputSize, firstLayerNeurons, outputSize, outputTypes, modelID, projectName)

		// Serialize the model to JSON
		modelFilePath := filepath.Join(modelDir, modelID+".json")
		modelFile, err := os.Create(modelFilePath)
		if err != nil {
			return fmt.Errorf("failed to create model file %s: %w", modelFilePath, err)
		}
		defer modelFile.Close()

		encoder := json.NewEncoder(modelFile)
		if err := encoder.Encode(modelConfig); err != nil {
			return fmt.Errorf("failed to serialize model %s: %w", modelFilePath, err)
		}

		log.Printf("Saved model %d to %s\n", i, modelFilePath)
	}

	return nil
}



// Load a model from a file
func loadModel(filePath string) (*dense.NetworkConfig, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	var modelConfig dense.NetworkConfig
	if err := json.NewDecoder(file).Decode(&modelConfig); err != nil {
		return nil, fmt.Errorf("failed to decode model: %w", err)
	}

	return &modelConfig, nil
}

// Save a model to a file
func saveModel(filePath string, modelConfig *dense.NetworkConfig) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()

	if err := json.NewEncoder(file).Encode(modelConfig); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	return nil
}

