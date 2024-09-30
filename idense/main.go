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
	//"math/rand"
	"os"
	"path/filepath"
	"sync"
	"strings"
	"time"
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


	// Load model and test data here
    modelConfig, err := loadModel("./host/generations/0/model_0.json")
    if err != nil {
        fmt.Println("Failed to load model:", err)
        return
    }

	 // Save sharded layer states for testing
	 fmt.Println("Saving sharded layer states...")
	 SaveShardedLayerStates(modelConfig, mnistData[:10], "./host/generations/0/model_0.json")
 
	 // Test performance of the model using sharded layer states
	 fmt.Println("Testing performance with sharded layer states...")
	 TestShardedModelPerformance(modelConfig, mnistData[:10], "./host/generations/0/model_0.json")

    //TestModelPerformance(modelConfig, mnistData, "./host/generations/0/model_0.json")
	//CompareModelOutputsWithLoadTimesSingleLoad(modelConfig, mnistData[:10], "./host/generations/0/model_0.json")
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

// SaveShardedLayerStates saves the layer states of the model in a sharded format.
// SaveShardedLayerStates saves the layer states of the model in a sharded format.
func SaveShardedLayerStates(modelConfig *dense.NetworkConfig, testData []MNISTImageData, modelFilePath string) {
    // Get the last hidden layer's index
    layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)

    for i, data := range testData {
        // Convert image to inputs
        inputs := convertImageToInputs(data.FileName)

        // Generate unique input ID (e.g., index of the test data)
        inputID := fmt.Sprintf("%d", i)

        // Call FeedforwardLayerStateSavingShard to process and save the layer states in shards
        dense.FeedforwardLayerStateSavingShard(modelConfig, inputs, layerStateNumber, modelFilePath, inputID)
    }

    fmt.Println("Sharded layer states have been saved.")
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
	accuracy := evaluateModel(trainData, modelConfig,modelFilePath)
	fmt.Println("accuracy",accuracy)
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

