package main

import (
	"dense"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	//"strconv"
)

// TopModel represents a model and its accuracy
type TopModel struct {
	Config   *dense.NetworkConfig
	Accuracy float64
	Path     string
}

var jsonFilePath string
var mnistData []dense.ImageData
var testDataChunk []dense.ImageData

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
	inputSize := 28 * 28 // Input size for MNIST data
	outputSize := 10     // Output size for MNIST digits (0-9)
	outputTypes := []string{
		"sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid",
		"sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid",
	} // Activation type for output layer

	//mnistDataFilePath := "./host/mnistData.json"
	//percentageTrain := 0.8
	numModels := 2
	generationNum := 50
	//projectPath := "./host/generations/"
	projectModels := "./host/models/"

	filesExist, _ := dense.FilesWithExtensionExistInCurrentFolder(projectModels, ".json")

	if filesExist {
		fmt.Println("Files with the specified extension already exist. Skipping model generation.")
	} else {
		fmt.Println("No files found with the specified extension. Generating models.")
		dense.GenerateModelsIfNotExist(projectModels, numModels, inputSize, outputSize, outputTypes, projectName)
	}

	testDataChunk = mnistData[:40000]

	percentageTrain := 0.8
	// Split the data into training and testing sets
	trainSize := int(percentageTrain * float64(len(mnistData)))
	trainData := mnistData[:trainSize]

	// Loop through trainData and assign the output map
	for i := range trainData {
		trainData[i].OutputMap = convertLabelToOutputMap(trainData[i].Label)
		//fmt.Println(trainData[i])
	}

	// Create a new slice of type []interface{}
	testDataInterface := make([]interface{}, len(trainData))

	// Convert each element from []dense.ImageData to []interface{}
	for i, data := range trainData {
		testDataInterface[i] = data
	}

	// Mutation types
	mutationTypes := []string{"AppendNewLayer", "AppendMultipleLayers", "AppendCNNAndDenseLayer", "AppendLSTMLayer"}

	// Define ranges for neurons/filters and layers dynamically
	neuronRange := [2]int{10, 128} // Min and max neurons or filters
	layerRange := [2]int{1, 5}     // Min and max layers

	//noImprovementCounter := 0

	for i := 0; i <= generationNum; i++ {

		dense.CreateModelShards(projectModels, &testDataInterface, mnistDir, i)

		dense.CreateLearnedOrNot(projectModels, &testDataInterface, mnistDir, i, true)

		dense.IncrementalLayerSearch(projectModels, &testDataInterface, i, mutationTypes, neuronRange, layerRange, 1000, true, 40, 5)
		//IncrementalLayerMutationSearch
		/*dense.EvaluateModelAccuracyFromLayerState(generationDir, &testDataInterface, mnistDir, true)

		// **Capture the return value of GenerateChildren**
		improvementsFound := dense.GenerateChildren(generationDir, &testDataInterface, mutationTypes, neuronRange, layerRange, 1000, true, 40)

		// **Update the noImprovementCounter based on improvementsFound**
		if improvementsFound {
			noImprovementCounter = 0 // Reset counter if improvements were found
		} else {
			noImprovementCounter++ // Increment counter if no improvements were found
		}

		// **Check if the counter has reached the threshold**
		if noImprovementCounter >= 5 {
			// Increase the neuron range
			neuronRange[1] += 10 // Increase the max neurons by 10 (adjust as needed)
			neuronRange[0] += 5
			fmt.Printf("No improvements for %d generations. Increasing neuronRange to: %v\n", noImprovementCounter, neuronRange)
			noImprovementCounter = 0 // Reset the counter after adjustment
		}

		dense.MoveChildrenToNextGeneration(generationDir, i, 100)*/
		//dense.DeleteAllFolders(generationDir)
		//CreateNextGeneration(generationDir, numModels, i)
		break
	}

	return

}

func convertLabelToOutputMap(label int) map[string]float64 {
	outputMap := make(map[string]float64)
	for i := 0; i < 10; i++ {
		outputMap[fmt.Sprintf("output%d", i)] = 0.0
	}
	outputMap[fmt.Sprintf("output%d", label)] = 1.0
	return outputMap
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

func LoadMNISTData() { // ([]dense.ImageData, error) {
	jsonFile, _ := os.Open(jsonFilePath)

	defer jsonFile.Close()

	byteValue, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		//return nil, err
	}

	//var mnistData []dense.ImageData
	err = json.Unmarshal(byteValue, &mnistData)
	if err != nil {
		//return nil, err
	}

	//return mnistData, nil
}
