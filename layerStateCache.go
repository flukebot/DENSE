// layerStateCache.go
package dense

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
)

// LayerHashInfo holds the hash and its position within the hidden layers.
type LayerHashInfo struct {
	Position int    `json:"position"`
	Hash     string `json:"hash"`
}

// LayerSequenceCount holds a layer sequence's hash and its occurrence count.
type LayerSequenceCount struct {
	Hash  string `json:"hash"`
	Count int    `json:"count"`
}

// GenerateLayerHashes processes the hidden layers of the network config,
// computes chained MD5 hashes, and returns an array of LayerHashInfo.
func GenerateLayerHashes(config *NetworkConfig) ([]LayerHashInfo, error) {
	var layerHashes []LayerHashInfo
	var cumulativeHash string

	for idx, layer := range config.Layers.Hidden {
		// Serialize the layer to JSON with consistent key ordering
		layerJSON, err := json.Marshal(layer)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal layer %d: %v", idx+1, err)
		}

		// Concatenate current layer JSON with all previous hashes
		var combined []byte
		combined = append(combined, layerJSON...)
		if cumulativeHash != "" {
			combined = append(combined, []byte(cumulativeHash)...)
		}

		// Compute MD5 hash
		hash := md5.Sum(combined)
		hashStr := hex.EncodeToString(hash[:])

		// Update cumulative hash by chaining
		if cumulativeHash == "" {
			cumulativeHash = hashStr
		} else {
			cumulativeHash += hashStr
		}

		// Append to the result
		layerHashes = append(layerHashes, LayerHashInfo{
			Position: idx + 1, // 1-based indexing
			Hash:     hashStr,
		})
	}

	return layerHashes, nil
}

// AnalyzeLayerHashes scans all .json files in the given directory,
// computes layer hashes, and returns the top N most common layer sequences.
func AnalyzeLayerHashes(folderPath string, topN int) ([]LayerSequenceCount, error) {
	// Map to count occurrences of each layer sequence hash
	hashCountMap := make(map[string]int)

	// Walk through the directory and process each .json file
	err := filepath.Walk(folderPath, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err // Abort if there's an error accessing the path
		}

		// Skip directories and non-.json files
		if info.IsDir() || filepath.Ext(path) != ".json" {
			return nil
		}

		// Load the network config from the JSON file
		config, err := LoadNetworkFromFile(path)
		if err != nil {
			fmt.Printf("Warning: failed to load config from %s: %v\n", path, err)
			return nil // Skip this file and continue
		}

		// Generate layer hashes
		layerHashes, err := GenerateLayerHashes(config)
		if err != nil {
			fmt.Printf("Warning: failed to generate hashes for %s: %v\n", path, err)
			return nil // Skip this file and continue
		}

		// Create a unique identifier for the layer sequence
		sequenceIdentifier := ""
		for _, lh := range layerHashes {
			sequenceIdentifier += fmt.Sprintf("%d:%s|", lh.Position, lh.Hash)
		}

		// Increment the count for this sequence
		hashCountMap[sequenceIdentifier]++

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("error walking through folder %s: %v", folderPath, err)
	}

	// Convert the map to a slice for sorting
	var layerSequenceCounts []LayerSequenceCount
	for seq, count := range hashCountMap {
		layerSequenceCounts = append(layerSequenceCounts, LayerSequenceCount{
			Hash:  seq,
			Count: count,
		})
	}

	// Sort the slice in descending order of count
	sort.Slice(layerSequenceCounts, func(i, j int) bool {
		return layerSequenceCounts[i].Count > layerSequenceCounts[j].Count
	})

	// Select the top N sequences
	if len(layerSequenceCounts) > topN {
		layerSequenceCounts = layerSequenceCounts[:topN]
	}

	return layerSequenceCounts, nil
}
