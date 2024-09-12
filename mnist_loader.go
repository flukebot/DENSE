package dense

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

// New URL to download MNIST dataset from Google's storage
const baseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

// DownloadFile downloads a file from a URL and saves it locally
func DownloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check if the status is 200 OK
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: %s, status code: %d", url, resp.StatusCode)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// EnsureMNISTDownloads ensures that the MNIST dataset is downloaded and unzipped correctly
func EnsureMNISTDownloads() error {
	// Updated file links from Google's storage
	files := []string{
		"train-images-idx3-ubyte.gz",
		"train-labels-idx1-ubyte.gz",
		"t10k-images-idx3-ubyte.gz",
		"t10k-labels-idx1-ubyte.gz",
	}

	for _, file := range files {
		localFile := file
		if _, err := os.Stat(localFile); os.IsNotExist(err) {
			log.Printf("Downloading %s...\n", file)
			if err := DownloadFile(localFile, baseURL+file); err != nil {
				return err
			}
			log.Printf("Downloaded %s\n", file)

			// Unzip the file
			if err := UnzipFile(localFile); err != nil {
				return err
			}
		} else {
			log.Printf("%s already exists, skipping download.\n", file)
		}
	}
	return nil
}

// UnzipFile extracts the MNIST dataset from .gz files
func UnzipFile(gzFile string) error {
	in, err := os.Open(gzFile)
	if err != nil {
		return err
	}
	defer in.Close()

	gz, err := gzip.NewReader(in)
	if err != nil {
		return fmt.Errorf("error creating gzip reader: %v", err)
	}
	defer gz.Close()

	outFile := gzFile[:len(gzFile)-3] // Remove .gz extension
	out, err := os.Create(outFile)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, gz)
	if err != nil {
		return fmt.Errorf("error during unzipping: %v", err)
	}

	log.Printf("Unzipped %s successfully\n", outFile)
	return nil
}

// MNISTData struct for storing MNIST images and labels
type MNISTData struct {
	Images [][]byte
	Labels []byte
}

// LoadMNIST loads the MNIST dataset from the unzipped files
func LoadMNISTOLD() (*MNISTData, error) {
	trainImages, err := loadImages("train-images-idx3-ubyte")
	if err != nil {
		return nil, fmt.Errorf("failed to load training images: %w", err)
	}

	trainLabels, err := loadLabels("train-labels-idx1-ubyte")
	if err != nil {
		return nil, fmt.Errorf("failed to load training labels: %w", err)
	}

	mnist := &MNISTData{
		Images: trainImages,
		Labels: trainLabels,
	}

	return mnist, nil
}

func SaveMNIST(filename string, mnist *MNISTData) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()

    encoder := json.NewEncoder(file)
    return encoder.Encode(mnist)
}

func LoadMNIST(filename string) (*MNISTData, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    mnist := new(MNISTData)
    decoder := json.NewDecoder(file)
    err = decoder.Decode(mnist)
    if err != nil {
        return nil, err
    }

    return mnist, nil
}

func loadImages(filename string) ([][]byte, error) {
	filePath := filepath.Join(".", filename)
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read magic number and dimensions
	var magicNumber, numImages, numRows, numCols int32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.BigEndian, &numRows); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.BigEndian, &numCols); err != nil {
		return nil, err
	}

	imageSize := int(numRows * numCols)
	images := make([][]byte, numImages)

	for i := 0; i < int(numImages); i++ {
		image := make([]byte, imageSize)
		if _, err := file.Read(image); err != nil {
			return nil, err
		}
		images[i] = image
	}

	return images, nil
}

func loadLabels(filename string) ([]byte, error) {
	filePath := filepath.Join(".", filename)
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read magic number and number of labels
	var magicNumber, numLabels int32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.BigEndian, &numLabels); err != nil {
		return nil, err
	}

	labels := make([]byte, numLabels)
	if _, err := file.Read(labels); err != nil {
		return nil, err
	}

	return labels, nil
}
