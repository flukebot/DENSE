package dense


import (
	"fmt"        // For fmt.Sprintf
	//"image"      // For image processing
	"image/color"// For color conversion
	"image/jpeg" // For decoding JPEG images
	"log"        // For logging errors
	"os"         // For opening files
)


func ConvertImageToInputs(filePath string) map[string]interface{} {
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