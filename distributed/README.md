# DENSE Distributed POC

## Distributed Evolutionary Network Search Engine - Proof of Concept

Welcome to the Distributed Proof of Concept (POC) for **DENSE (Distributed Evolutionary Network Search Engine)**. This component demonstrates the distributed capabilities of DENSE, focusing on secure communication and efficient neural network architecture exploration using sharding and encryption.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Security](#security)
- [Sharding Mechanism](#sharding-mechanism)


## Overview

DENSE is a Golang-based neural network library designed for flexibility and scalability. This distributed POC focuses on enabling secure and efficient distribution of neural network tasks across multiple clients. It leverages WebSockets with robust encryption to ensure secure communication between the server and clients, preventing attacks such as SSL stripping.

## Features

- **Secure WebSocket Communication**: Implements end-to-end encryption using ECDH key exchange and AES-GCM to safeguard data transmission.
- **Layer Sharding**: Distributes neural network layers across multiple clients for parallel processing, enhancing scalability and reducing computational overhead.
- **Real-Time Collaboration**: Allows multiple nodes to collaborate in evaluating and refining neural network models in real-time.
- **Dynamic Evolution**: Continuously evolves neural network architectures based on feedback from distributed training sessions.
- **WebAssembly Support**: Runs both natively and as WebAssembly, enabling compatibility with various environments, including web browsers.

## Architecture

### Secure Communication

To ensure secure communication and prevent SSL stripping, the distributed POC employs the following encryption strategy:

1. **Key Exchange**: Utilizes Elliptic Curve Diffie-Hellman (ECDH) to establish a shared secret between the server and client.
2. **Symmetric Encryption**: Derives a symmetric AES-256 key from the shared secret using SHA-256 hashing.
3. **AES-GCM Encryption**: Encrypts all WebSocket messages with AES-GCM, providing confidentiality and integrity.

### Sharding Mechanism

The system employs **Layer Sharding** to split neural network layers into shards, which are then distributed to clients for processing. This approach allows:

- **Parallel Processing**: Multiple clients can process different shards simultaneously, accelerating training and evaluation.
- **Scalability**: Easily scales with the number of available clients, optimizing resource utilization.
- **Fault Tolerance**: Distributes the computational load, reducing the impact of individual client failures.

## Security

Security is a paramount concern in distributed systems. This POC ensures secure data transmission through:

- **End-to-End Encryption**: All data exchanged between the server and clients is encrypted, preventing eavesdropping and man-in-the-middle attacks.
- **Authentication**: Clients must authenticate using a secure password mechanism before participating in the distributed training process.
- **Key Management**: Securely generates and manages ECDSA key pairs for each server instance, ensuring unique and robust encryption keys.

## Sharding Mechanism

Sharding in DENSE involves dividing the neural network into smaller, manageable layers or sections called shards. Each shard is assigned to a client, which performs computations and sends back results. The server coordinates these shards, aggregates the results, and evolves the network architecture based on performance metrics.

### Benefits

- **Efficiency**: Reduces the time required for training by leveraging multiple clients.
- **Flexibility**: Allows for dynamic allocation and reallocation of shards based on client performance and availability.
- **Scalability**: Easily accommodates additional clients to handle increased computational demands.

