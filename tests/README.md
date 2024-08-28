# Tests for CortexBuilder

This folder contains the test suite for the CortexBuilder project. The tests are designed to validate the functionality of the neural network implementation, including its ability to correctly configure networks, perform feedforward operations, and optimize network parameters using random hill climbing.

## Overview of Tests

### 1. **Network Configuration Testing**

The tests validate that the network configuration, including input, hidden, and output layers, is correctly set up. This includes ensuring that neurons are properly connected, and activation functions are correctly applied.

### 2. **Feedforward Propagation Testing**

The feedforward propagation tests ensure that the network can correctly process input data and produce the expected outputs. This involves testing various configurations of the network to ensure that the feedforward function operates correctly under different scenarios.

### 3. **Random Hill Climbing Optimization**

These tests focus on the network's ability to improve its performance through random mutations. The optimization process involves:
- Randomly mutating the network's weights, biases, and structure.
- Evaluating the network's performance after each mutation.
- Logging improvements and saving the network configuration when a better-performing model is found.

### 4. **Performance Logging**

Throughout the optimization process, performance improvements are logged. This includes details such as the iteration number, the error before and after mutation, and the percentage of improvement.

## Running the Tests

To run the tests, navigate to the `tests` directory and execute the following command:

```bash
go run main.go
