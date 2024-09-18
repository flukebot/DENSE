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

func main() {
	fmt.Println("Starting CNN train and host")

	// Check if the MNIST directory exists, and run setup if it doesn't
	mnistDir := "./host/MNIST"
	if !dense.CheckDirExists(mnistDir) {
		fmt.Println("MNIST directory doesn't exist, running setupMNIST()")
		setupMNIST()
	} else {
		fmt.Println("MNIST directory already exists, skipping setup.")
	}

	// Set up the model configuration
	projectName := "AIModelTestProject"
	inputSize := 28 * 28   // Input size for MNIST data
	outputSize := 10       // Output size for MNIST digits (0-9)
	outputTypes := []string{"softmax"} // Activation type for output layer
	mnistDataFilePath := "./host/mnistData.json"
	percentageTrain := 0.8
	numModels := 100
	generationNum := 500
	/*modelConfig := dense.CreateRandomNetworkConfig(inputSize, outputSize, outputTypes, "id1", projectName)


	
	// Define the path to the MNIST data JSON file
	jsonFilePath := "./host/mnistData.json"

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







	
	// Loop from 0 to generationNum
	for i := 0; i <= generationNum; i++ {

		// Define the directories for the current and next generation
		currentGenDir := fmt.Sprintf("./host/generations/%d", i)
		nextGenDir := fmt.Sprintf("./host/generations/%d", i+1) // Increment for next generation

		// Print the current generation number
		fmt.Printf("Processing generation: %d\n", i)

		// Call the mutate function to mutate models inside the current generation
		fmt.Printf("Mutating models in generation %d...\n", i)
		err := MutateAllModelsRandomly(currentGenDir, inputSize, outputSize, outputTypes, projectName, mnistDataFilePath, percentageTrain,true)
		if err != nil {
			log.Fatalf("Error mutating models in generation %d: %v", i, err)
		}
		fmt.Printf("All models in generation %d have been mutated, evaluated, and saved.\n", i)

		// Evolve to the next generation
		fmt.Printf("Evolving models from generation %d to %d...\n", i, i+1)
		err = EvolveNextGeneration(currentGenDir, nextGenDir, numModels, 0.1, inputSize, outputSize, outputTypes, projectName, mnistDataFilePath, percentageTrain)
		if err != nil {
			log.Fatalf("Error evolving models from generation %d to %d: %v", i, i+1, err)
		}
		fmt.Printf("Successfully evolved generation %d to %d.\n", i, i+1)
	}

	fmt.Println("Completed all generations.")

}


//step 1-----------------------------

// LoadMNISTData loads the MNIST data from the JSON file and returns an array of MNISTImageData
func LoadMNISTData(jsonFilePath string) ([]MNISTImageData, error) {
	jsonFile, err := os.Open(jsonFilePath)
	if err != nil {
		return nil, err
	}
	defer jsonFile.Close()

	byteValue, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		return nil, err
	}

	var mnistData []MNISTImageData
	err = json.Unmarshal(byteValue, &mnistData)
	if err != nil {
		return nil, err
	}

	return mnistData, nil
}


func setupMNIST(){
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


//step 2------------------------------
// TrainAndEvaluateModel trains the model using a specified percentage of the data, then evaluates it
func EvaluateModel(jsonFilePath string, modelConfig *dense.NetworkConfig, percentageTrain float64) (float64, error) {
	// Load MNIST data from JSON file
	mnistData, err := LoadMNISTData(jsonFilePath)
	if err != nil {
		return 0.0, fmt.Errorf("failed to load MNIST data: %w", err)
	}

	// Shuffle the data to ensure randomness
	rand.Shuffle(len(mnistData), func(i, j int) {
		mnistData[i], mnistData[j] = mnistData[j], mnistData[i]
	})

	// Split the data into training and testing sets
	trainSize := int(percentageTrain * float64(len(mnistData)))
	trainData := mnistData[:trainSize]
	//testData := mnistData[trainSize:]

	// Evaluate the model on the testing data
	fmt.Println("Evaluating the model...")
	accuracy := evaluateModel(trainData, modelConfig)

	return accuracy, nil
}


// evaluateModel evaluates the model using the provided test data and returns accuracy
func evaluateModel(testData []MNISTImageData, modelConfig *dense.NetworkConfig) float64 {
	correct := 0
	for _, data := range testData {
		inputs := convertImageToInputs(data.FileName) // Function to convert image to input values
		outputPredicted := dense.Feedforward(modelConfig, inputs)

		// Get the predicted label (the index of the max output value)
		predictedLabel := getMaxIndex(outputPredicted)

		// Compare with the actual label
		if predictedLabel == data.Label {
			correct++
		}
	}

	// Calculate accuracy
	accuracy := float64(correct) / float64(len(testData))
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
		if outputs[fmt.Sprintf("output%d", i)] > maxValue {
			maxIndex = i
			maxValue = outputs[fmt.Sprintf("output%d", i)]
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
		modelConfig := dense.CreateRandomNetworkConfig(inputSize, outputSize, outputTypes, modelID, projectName)

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


//step 4 randomly mutate ---------------
// MutateAllModelsRandomly mutates all models inside the given generation directory.
// If `useGoroutines` is true, it will process each model in batches of 10 with concurrent goroutines.
func MutateAllModelsRandomly(generationDir string, inputSize, outputSize int, outputTypes []string, projectName string, mnistDataFilePath string, percentageTrain float64, useGoroutines bool) error {
	// Targeting the `generationDir` folder
	files, err := ioutil.ReadDir(generationDir)
	if err != nil {
		return fmt.Errorf("failed to read models directory: %w", err)
	}

	totalFiles := len(files)
	batchSize := 10 // Number of models to process concurrently
	batchCount := (totalFiles / batchSize) + 1

	var wg sync.WaitGroup

	for batchIndex := 0; batchIndex < batchCount; batchIndex++ {
		// Create a batch of files to process
		startIndex := batchIndex * batchSize
		endIndex := startIndex + batchSize
		if endIndex > totalFiles {
			endIndex = totalFiles
		}
		batchFiles := files[startIndex:endIndex]

		// Process the current batch
		for _, file := range batchFiles {
			if filepath.Ext(file.Name()) == ".json" {
				modelFilePath := filepath.Join(generationDir, file.Name())

				// If goroutines are enabled, process each file in a separate thread
				if useGoroutines {
					wg.Add(1)
					go func(modelFilePath string) {
						defer wg.Done()
						if err := processModel(modelFilePath, mnistDataFilePath, percentageTrain); err != nil {
							log.Printf("Error processing model %s: %v", modelFilePath, err)
						}
					}(modelFilePath)
				} else {
					// Process files sequentially without goroutines
					if err := processModel(modelFilePath, mnistDataFilePath, percentageTrain); err != nil {
						log.Printf("Error processing model %s: %v", modelFilePath, err)
					}
				}
			}
		}

		// Wait for all goroutines in the batch to complete
		if useGoroutines {
			wg.Wait()
		}

		// Calculate and print progress percentage
		progress := float64(endIndex) / float64(totalFiles) * 100
		fmt.Printf("Batch %d/%d processed, %.2f%% complete.\n", batchIndex+1, batchCount, progress)
	}

	return nil
}

// processModel handles loading, mutating, evaluating, and saving a single model.
func processModel(modelFilePath, mnistDataFilePath string, percentageTrain float64) error {
	// Load the model
	modelConfig, err := loadModel(modelFilePath)
	if err != nil {
		return fmt.Errorf("failed to load model %s: %w", modelFilePath, err)
	}

	// Check if the model has already been evaluated
	if modelConfig.Metadata.LastTestAccuracy > 0 {
		log.Printf("Model %s has already been evaluated with accuracy: %.2f%%. Skipping mutation and evaluation.", modelFilePath, modelConfig.Metadata.LastTestAccuracy*100)
		return nil
	}

	// Apply a random mutation to the model
	applyRandomMutation(modelConfig)

	// Evaluate the model after mutation
	accuracy, err := EvaluateModel(mnistDataFilePath, modelConfig, percentageTrain)
	if err != nil {
		return fmt.Errorf("failed to evaluate model %s: %w", modelFilePath, err)
	}

	// Update the metadata with the new accuracy
	modelConfig.Metadata.LastTestAccuracy = accuracy

	// Save the mutated model back to the same file
	if err := saveModel(modelFilePath, modelConfig); err != nil {
		return fmt.Errorf("failed to save mutated model %s: %w", modelFilePath, err)
	}

	log.Printf("Mutated and evaluated model saved: %s (Accuracy: %.2f%%)", modelFilePath, accuracy*100)
	return nil
}


// Apply a random number of mutations to a model
func applyRandomMutation(config *dense.NetworkConfig) {
    mutations := []func(*dense.NetworkConfig){
        func(c *dense.NetworkConfig) { dense.MutateCNNWeights(c, 0.1, 40) },
        func(c *dense.NetworkConfig) { dense.MutateCNNBiases(c, 40, 0.1) },
        func(c *dense.NetworkConfig) { dense.RandomizeCNNWeights(c, 20) },
        func(c *dense.NetworkConfig) { dense.InvertCNNWeights(c, 40) },
        func(c *dense.NetworkConfig) { dense.AddCNNLayerAtRandomPosition(c, 40) },
        func(c *dense.NetworkConfig) { dense.MutateCNNFilterSize(c, 40) },           // Add mutation to CNN filter size
        func(c *dense.NetworkConfig) { dense.MutateCNNStrideAndPadding(c, 40) },     // Add mutation to CNN stride and padding
        func(c *dense.NetworkConfig) { dense.DuplicateCNNLayer(c, 40) },             // Add mutation to duplicate CNN layers
        func(c *dense.NetworkConfig) { dense.AddMultipleCNNLayers(c, 40, 5) },       // Add multiple CNN layers
    }

    // Select a random number of mutations between 1 and 30
    rand.Seed(time.Now().UnixNano())
    numMutations := rand.Intn(30) + 1 // Random number from 1 to 30

    // Apply the random number of mutations
    for i := 0; i < numMutations; i++ {
        mutation := mutations[rand.Intn(len(mutations))] // Randomly select a mutation
        mutation(config)
    }

    fmt.Printf("Applied %d mutations to the model.\n", numMutations)
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


//step 5 --------------
//step 5 --------------
func EvolveNextGeneration(currentGenDir, nextGenDir string, numModels int, topPercentage float64, inputSize, outputSize int, outputTypes []string, projectName, mnistDataFilePath string, percentageTrain float64) error {
	// Read all models from the current generation
	files, err := ioutil.ReadDir(currentGenDir)
	if err != nil {
		return fmt.Errorf("failed to read current generation directory: %w", err)
	}

	var models []*dense.NetworkConfig

	// Load all models
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".json" {
			modelFilePath := filepath.Join(currentGenDir, file.Name())

			modelConfig, err := loadModel(modelFilePath)
			if err != nil {
				log.Printf("Failed to load model %s: %v", modelFilePath, err)
				continue
			}

			// Evaluate the model if not already evaluated
			if modelConfig.Metadata.LastTestAccuracy == 0 {
				accuracy, err := EvaluateModel(mnistDataFilePath, modelConfig, percentageTrain)
				if err != nil {
					log.Printf("Failed to evaluate model %s: %v", modelFilePath, err)
					continue
				}
				modelConfig.Metadata.LastTestAccuracy = accuracy
				saveModel(modelFilePath, modelConfig)
			}

			// Add the model to the list
			models = append(models, modelConfig)
		}
	}

	// Sort the models by accuracy (descending)
	models = findTopModels(models, topPercentage)

	// Create the directory for the next generation
	if err := os.MkdirAll(nextGenDir, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create next generation directory: %w", err)
	}

	// Number of top models to select
	topCount := int(float64(numModels) * topPercentage)

	// 1. Copy the top 10% without mutations to the next generation
	for i := 0; i < topCount; i++ {
		modelID := fmt.Sprintf("model_%d", i)
		modelFilePath := filepath.Join(nextGenDir, modelID+".json")

		// Load the model from the current generation
		modelConfig := models[i]

		  // Reset the LastTestAccuracy
		  modelConfig.Metadata.LastTestAccuracy = 0.0

		// Save the model without any mutations in the next generation folder
		if err := saveModel(modelFilePath, modelConfig); err != nil {
			return fmt.Errorf("failed to save model: %v", err)
		}
	}

	// 2. Calculate how many models each top model will generate with mutations
	mutationCountPerTopModel := (numModels - topCount) / topCount

	// 3. Mutate the top models and fill the remaining spots in the next generation
	index := topCount
	for i := 0; i < topCount; i++ {
		for j := 0; j < mutationCountPerTopModel; j++ {
			// Reload the top model from the current generation
			modelConfig := models[i]

			// Apply random mutation
			applyRandomMutation(modelConfig)

			// Save the mutated model into the next generation folder
			modelID := fmt.Sprintf("model_%d", index)
			modelFilePath := filepath.Join(nextGenDir, modelID+".json")
			modelConfig.Metadata.LastTestAccuracy = 0.0
			if err := saveModel(modelFilePath, modelConfig); err != nil {
				return fmt.Errorf("failed to save mutated model: %v", err)
			}

			index++
		}
	}

	// If we have any remaining models to fill (because of rounding issues)
	for index < numModels {
		// Reload additional models from the top ones
		modelConfig := models[index % topCount]

		// Apply random mutation
		applyRandomMutation(modelConfig)

		// Save the model with mutation to the next generation
		modelID := fmt.Sprintf("model_%d", index)
		modelFilePath := filepath.Join(nextGenDir, modelID+".json")
		if err := saveModel(modelFilePath, modelConfig); err != nil {
			return fmt.Errorf("failed to save mutated model: %v", err)
		}

		index++
	}

	log.Printf("Successfully created generation %s with %d models.", nextGenDir, numModels)
	return nil
}


// Helper function to find the top models based on accuracy
func findTopModels(models []*dense.NetworkConfig, topPercentage float64) []*dense.NetworkConfig {
	// Sort models based on LastTestAccuracy (descending order)
	for i := 0; i < len(models); i++ {
		for j := i + 1; j < len(models); j++ {
			if models[i].Metadata.LastTestAccuracy < models[j].Metadata.LastTestAccuracy {
				models[i], models[j] = models[j], models[i]
			}
		}
	}

	// Return the top percentage of models
	topCount := int(float64(len(models)) * topPercentage)
	return models[:topCount]
}



func OLDmain() {
	rand.Seed(time.Now().UnixNano())

	// Test 1: FFNN + CNN Model
	fmt.Println("Test 1: FFNN + CNN Model")

	// Create a model configuration with FFNN and CNN layers
	config1 := dense.CreateRandomNetworkConfig(3, 1, []string{"relu"}, "ffnn_cnn_test", "FFNN + CNN Test")
	addConvLayer(config1)
	addDenseLayer(config1)

	// Apply CNN mutations
	applyCNNMutations(config1)

	// Test 2: CNN + LSTM Model
	fmt.Println("\nTest 2: CNN + LSTM Model")

	// Create a model configuration with CNN and LSTM layers
	config2 := dense.CreateRandomNetworkConfig(3, 1, []string{"relu"}, "cnn_lstm_test", "CNN + LSTM Test")
	addConvLayer(config2)
	addLSTMLayer(config2)

	// Apply CNN mutations
	applyCNNMutations(config2)

	// Test 3: CNN + LSTM + CNN Model
	fmt.Println("\nTest 3: CNN + LSTM + CNN Model")

	// Create a model configuration with CNN, LSTM, and CNN layers
	config3 := dense.CreateRandomNetworkConfig(3, 1, []string{"relu"}, "cnn_lstm_cnn_test", "CNN + LSTM + CNN Test")
	addConvLayer(config3)
	addLSTMLayer(config3)
	addConvLayer(config3)

	// Apply CNN mutations
	applyCNNMutations(config3)
}

// Adds a CNN layer to the config
func addConvLayer(config *dense.NetworkConfig) {
	newLayer := dense.Layer{
		LayerType: "conv",
		Filters: []dense.Filter{
			{
				Weights: dense.Random2DSlice(3, 3),
				Bias:    rand.Float64(),
			},
		},
		Stride:  1,
		Padding: 1,
	}

	config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
}

// Adds an LSTM layer to the config
func addLSTMLayer(config *dense.NetworkConfig) {
	newLayer := dense.Layer{
		LayerType: "lstm",
		LSTMCells: []dense.LSTMCell{
			{
				InputWeights:  dense.RandomSlice(3),
				ForgetWeights: dense.RandomSlice(3),
				OutputWeights: dense.RandomSlice(3),
				CellWeights:   dense.RandomSlice(3),
				Bias:          rand.Float64(),
			},
		},
	}

	config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
}

// Adds a Dense layer to the config
func addDenseLayer(config *dense.NetworkConfig) {
	newLayer := dense.Layer{
		LayerType: "dense",
		Neurons: map[string]dense.Neuron{
			"hidden1": {
				ActivationType: "relu",
				Connections: map[string]dense.Connection{
					"input0": {Weight: rand.Float64()},
					"input1": {Weight: rand.Float64()},
					"input2": {Weight: rand.Float64()},
				},
				Bias: rand.Float64(),
			},
		},
	}

	config.Layers.Hidden = append(config.Layers.Hidden, newLayer)
}

// Apply CNN mutations to the network
func applyCNNMutations(config *dense.NetworkConfig) {
	fmt.Println("Applying MutateCNNWeights...")
	dense.MutateCNNWeights(config, 0.01, 20)

	printCNNLayer(config)

	fmt.Println("Applying MutateCNNBiases...")
	dense.MutateCNNBiases(config, 20, 0.01)

	printCNNLayer(config)

	fmt.Println("Applying RandomizeCNNWeights...")
	dense.RandomizeCNNWeights(config, 20)

	printCNNLayer(config)

	fmt.Println("Applying InvertCNNWeights...")
	dense.InvertCNNWeights(config, 20)

	printCNNLayer(config)

	fmt.Println("Applying AddCNNLayerAtRandomPosition...")
	dense.AddCNNLayerAtRandomPosition(config, 20)

	printCNNLayer(config)
}

// Helper function to print CNN layer details
func printCNNLayer(config *dense.NetworkConfig) {
	for _, layer := range config.Layers.Hidden {
		if layer.LayerType == "conv" {
			fmt.Println("CNN Layer:")
			for i, filter := range layer.Filters {
				fmt.Printf("  Filter %d:\n", i)
				fmt.Printf("    Weights:  %v\n", filter.Weights)
				fmt.Printf("    Bias:     %v\n", filter.Bias)
			}
		}
	}
}
