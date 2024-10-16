package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"dense"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"github.com/gorilla/websocket"
	"github.com/joho/godotenv"
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
var envPWD string

var (
	upgrader      = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	serverPrivKey *ecdsa.PrivateKey
	serverPubPEM  []byte
)

func main() {
	fmt.Println("----------------dense----------")

	// Load environment variables
	err := godotenv.Load()
	if err != nil {
		fmt.Println("Error loading .env file")
	}

	envPort := os.Getenv("PORT")
	if envPort == "" {
		envPort = "4125"
	}

	envPWD = os.Getenv("SERVERPWD")
	if envPWD == "" {
		envPWD = "securepassword"
	}

	// Generate server's ECDSA key pair
	serverPrivKey, serverPubPEM, err = dense.GenerateKeyPair()
	if err != nil {
		log.Fatalf("Failed to generate server key pair: %v", err)
	}

	// HTTP endpoint to retrieve the server's public key (optional)
	http.HandleFunc("/publickey", handlePublicKey)

	// WebSocket handler
	http.HandleFunc("/ws", handleConnections)

	// Start HTTP server
	log.Printf("Server started on :%s", envPort)
	log.Fatal(http.ListenAndServe(":"+envPort, nil))
	/*jsonFilePath = "./host/mnistData.json"
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
	generationNum := 500
	projectPath := "./host/generations/"

	filesExist, _ := dense.FilesWithExtensionExistInCurrentFolder(projectPath+"0", ".json")


	if filesExist {
		fmt.Println("Files with the specified extension already exist. Skipping model generation.")
	} else {
		fmt.Println("No files found with the specified extension. Generating models.")
		dense.GenerateModelsIfNotExist(projectPath+"0", numModels, inputSize, outputSize, outputTypes, projectName)
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

	noImprovementCounter := 0

	for i := 0; i <= generationNum; i++ {
		generationDir := "./host/generations/" + strconv.Itoa(i)
		fmt.Println("----CURENT GEN---", generationDir)

		dense.SaveLayerStates(generationDir, &testDataInterface, mnistDir)
		dense.EvaluateModelAccuracyFromLayerState(generationDir, &testDataInterface, mnistDir, true)

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

		dense.MoveChildrenToNextGeneration(generationDir, i, 100)
		//dense.DeleteAllFolders(generationDir)
		//CreateNextGeneration(generationDir, numModels, i)
		//break
	}

	return*/

}

// Helper function to convert label to a one-hot encoded map with float64 values
func convertLabelToOutputMap(label int) map[string]float64 {
	outputMap := make(map[string]float64)
	for i := 0; i < 10; i++ {
		outputMap[fmt.Sprintf("output%d", i)] = 0.0
	}
	outputMap[fmt.Sprintf("output%d", label)] = 1.0
	return outputMap
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

// Handle /publickey endpoint (optional)
func handlePublicKey(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/x-pem-file")
	w.Write(serverPubPEM)
}

func handleConnections(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Step 1: Send server's public key to client
	log.Printf("Sending server's public key to client %s:\n%s", conn.RemoteAddr(), string(serverPubPEM))
	if err := conn.WriteMessage(websocket.TextMessage, serverPubPEM); err != nil {
		log.Printf("Error sending public key: %v", err)
		return
	}

	// Step 2: Receive client's public key
	_, clientPubKeyPEM, err := conn.ReadMessage()
	if err != nil {
		log.Printf("Error reading client's public key: %v", err)
		return
	}

	// Decode client's public key
	clientPubKey, err := parsePublicKey(clientPubKeyPEM)
	if err != nil {
		log.Printf("Error parsing client's public key: %v", err)
		return
	}

	// Step 3: Derive shared secret using ECDH
	sharedSecret, err := deriveSharedSecret(serverPrivKey, clientPubKey)
	if err != nil {
		log.Printf("Error deriving shared secret: %v", err)
		return
	}

	// Pad sharedSecret to 32 bytes
	if len(sharedSecret) < 32 {
		padded := make([]byte, 32)
		copy(padded[32-len(sharedSecret):], sharedSecret)
		sharedSecret = padded
	} else if len(sharedSecret) > 32 {
		// Shouldn't happen for P-256
		log.Printf("Shared secret longer than expected: %d bytes", len(sharedSecret))
		sharedSecret = sharedSecret[:32]
	}

	// Derive symmetric key from shared secret
	symmetricKey := sha256.Sum256(sharedSecret) // 32-byte key for AES-256

	log.Printf("Symmetric key established with client %s", conn.RemoteAddr())

	// Initialize AES-GCM cipher
	aesCipher, err := aes.NewCipher(symmetricKey[:])
	if err != nil {
		log.Printf("Error creating AES cipher: %v", err)
		return
	}
	gcm, err := cipher.NewGCM(aesCipher)
	if err != nil {
		log.Printf("Error creating GCM: %v", err)
		return
	}

	nonceSize := gcm.NonceSize()

	// Communication Phase
	for {
		_, encryptedMessage, err := conn.ReadMessage()
		if err != nil {
			log.Printf("Read error: %v", err)
			break
		}

		if len(encryptedMessage) < nonceSize {
			log.Printf("Invalid message size from client %s", conn.RemoteAddr())
			continue
		}

		nonce, ciphertext := encryptedMessage[:nonceSize], encryptedMessage[nonceSize:]
		decryptedMessage, err := gcm.Open(nil, nonce, ciphertext, nil)
		if err != nil {
			log.Printf("Decryption error: %v", err)
			continue
		}

		log.Printf("Received: %s", string(decryptedMessage))

		// Process the decrypted message
		var incomingMessage map[string]interface{}
		if err := json.Unmarshal(decryptedMessage, &incomingMessage); err != nil {
			log.Printf("JSON unmarshal error: %v", err)
			continue
		}

		// Example: Handle 'ping' message
		if incomingMessage["type"] == "ping" {
			// Verify password if necessary
			// For example, you can include password verification in the message
			if msgPwd, ok := incomingMessage["password"].(string); ok && msgPwd == envPWD {
				response := map[string]string{"msgType": "pong", "data": "Server response"}
				responseJSON, _ := json.Marshal(response)

				// Encrypt response
				nonce = make([]byte, nonceSize)
				if _, err := rand.Read(nonce); err != nil {
					log.Printf("Failed to generate nonce: %v", err)
					continue
				}
				encryptedResponse := gcm.Seal(nonce, nonce, responseJSON, nil)

				// Send encrypted response
				if err := conn.WriteMessage(websocket.BinaryMessage, encryptedResponse); err != nil {
					log.Printf("Write error: %v", err)
					break
				}
			} else {
				log.Printf("Incorrect password from client %s", conn.RemoteAddr())
				// Optionally send an error message before closing
				response := map[string]string{"msgType": "auth_error", "data": "Authentication failed"}
				responseJSON, _ := json.Marshal(response)
				nonce = make([]byte, nonceSize)
				if _, err := rand.Read(nonce); err == nil {
					encryptedResponse := gcm.Seal(nonce, nonce, responseJSON, nil)
					conn.WriteMessage(websocket.BinaryMessage, encryptedResponse)
				}
				conn.Close()
				return
			}
		}

		// Handle other message types as needed
	}
}

func parsePublicKey(pemBytes []byte) (*ecdsa.PublicKey, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil || block.Type != "PUBLIC KEY" {
		return nil, errors.New("failed to decode PEM block containing public key")
	}

	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	ecdsaPub, ok := pub.(*ecdsa.PublicKey)
	if !ok {
		return nil, errors.New("public key is not ECDSA")
	}

	return ecdsaPub, nil
}

func deriveSharedSecret(priv *ecdsa.PrivateKey, pub *ecdsa.PublicKey) ([]byte, error) {
	if priv.Curve != pub.Curve {
		return nil, errors.New("public key is not on the same curve as private key")
	}
	x, _ := priv.Curve.ScalarMult(pub.X, pub.Y, priv.D.Bytes())
	sharedSecret := x.Bytes()
	return sharedSecret, nil
}
