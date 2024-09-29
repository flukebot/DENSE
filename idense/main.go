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
	//"time"
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






func evaluateModel(testData []MNISTImageData, modelConfig *dense.NetworkConfig,modelFilePath string) float64 {
    correct := 0

	layerStateNumber := dense.GetLastHiddenLayerIndex(modelConfig)
	//fmt.Println("layerStateNumber",layerStateNumber)

    for _, data := range testData {
		//fmt.Println(index)
        inputs := convertImageToInputs(data.FileName) // Convert image to input values
        outputPredicted := dense.FeedforwardLayerStateSaving(modelConfig, inputs,layerStateNumber,modelFilePath) // Get predicted outputs

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

