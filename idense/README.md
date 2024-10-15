# AI Model Training Project

This project implements a convolutional neural network (CNN) with support for various model mutations and evolutionary training using the MNIST dataset. It automates model training, mutation, and generation of child models to optimize the CNN architecture.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Code Structure](#code-structure)
- [Detailed Workflow](#detailed-workflow)
- [Mutation Types](#mutation-types)
- [Handling No Improvements](#handling-no-improvements)
- [File Operations](#file-operations)
- [Getting Started](#getting-started)

## Overview

The primary goal of this project is to train a CNN on the MNIST dataset using evolutionary techniques to generate new models based on mutations. The system tracks improvements in accuracy and adjusts the network's architecture when no significant improvements are found for several generations.

## Key Features

- **Model Training and Evaluation**: Automatically trains and evaluates models on the MNIST dataset.
- **Model Mutation**: Generates new models through various mutation strategies.
- **Sharding**: Data is processed in shards for parallel training.
- **Automatic Model Generation**: If models don't exist, the system generates new models.
- **No Improvement Handling**: If no improvements are found over several generations, the neuron count is increased dynamically.
- **File Persistence**: Models and data states are saved to disk and loaded when needed.

## Code Structure

### `main.go`

This file contains the main entry point of the application and controls the flow of the training and evaluation process.

- **Main Logic Flow**:
  1. Initializes the project by checking if the MNIST dataset exists and loads it.
  2. Configures the CNN model parameters.
  3. Checks for pre-existing models, and if not found, generates new models.
  4. Loads MNIST data, sharding it for parallel processing.
  5. Trains the models on the shards and evaluates the accuracy.
  6. Applies mutations to models if improvements are not found.

### `dense.go`

This file defines the network structure, layers (dense, convolutional, LSTM), and mutation strategies. Key functionalities include:

- **NetworkConfig**: Defines the structure of the CNN, including input, hidden, and output layers.
- **Feedforward**: Implements feedforward logic for model evaluation.
- **Mutations**: Handles different mutation strategies (e.g., appending layers, CNN, and LSTM mutations).
- **Layer State Saving**: Saves intermediate layer states during model training.
- **Sharding**: Breaks the dataset into smaller parts for parallel processing.

## Detailed Workflow

1. **Start and Setup**: The process begins by initializing the project and checking whether the MNIST dataset exists in the `./host/MNIST` directory. If not, it runs `setupMNIST()` to download and prepare the dataset.
   
2. **Load Data**: Loads the MNIST data from JSON and configures model parameters such as input size (28x28 for MNIST), output size (10 for digits 0-9), and activation types (sigmoid).

3. **Model Generation**: If no existing models are found in `./host/generations/`, the system generates models with a specified number of neurons and layers.

4. **Training and Evaluation**:
   - The data is split into training and testing sets.
   - The models are trained on the shards of data, and their layer states are saved.
   - The models' performance is evaluated based on their accuracy.

5. **Mutation and Child Model Generation**: 
   - If no improvements are found, the system applies different mutations to the model structure. These include adding new layers, CNN and dense layer combinations, or LSTM layers.
   - Child models are generated and evaluated for improvements.
   
6. **Handling No Improvements**: If no improvements are found after five generations, the system dynamically increases the neuron range for the models, allowing for larger and potentially more complex networks.

7. **Saving Models**: After training and evaluation, the models' state and layer data are saved to disk for future use.

## Mutation Types

The system implements several mutation strategies to evolve the model's architecture:

- **AppendNewLayer**: Adds a new dense layer with a random number of neurons.
- **AppendMultipleLayers**: Adds multiple dense layers with varying neuron counts.
- **AppendCNNAndDenseLayer**: Adds a combination of convolutional layers and dense layers.
- **AppendLSTMLayer**: Adds an LSTM layer to the network to handle sequence-based data.

## Handling No Improvements

A key feature of the system is adjusting the model architecture when no improvements are found after several generations. The system tracks improvements, and if none are found after 5 iterations:
- The neuron count is increased by 10.
- The minimum neuron count is increased by 5.

This helps the model grow dynamically and search for a better configuration.

## File Operations

- **Model Saving**: Model configurations and layer states are saved as JSON files in the `./host/generations/` directory.
- **Shard Saving**: Intermediate layer states are saved for each shard of data during training to enable incremental evaluation.
- **Model Loading**: If existing models are found, they are loaded from disk for continued training and mutation.

## Getting Started

### Prerequisites

- Install Go (latest version)
- Download the MNIST dataset (automated in `setupMNIST()`)

### Running the Project

1. Clone the repository.
2. Run the following command to start the training process:
   ```bash
   go run main.go
```

## Diagram

For a detailed visual representation of the workflow, mutations, and file handling, please refer to the following diagram:

![GRAPH](./diagram.jpg)