# DENSE (Distributed Evolutionary Network Search Engine)

This section explains the **single-machine approach** for DENSE before rebuilding the functionality for distributed computing. The single-machine setup focuses on creating a baseline model and optimizing it generation by generation, while laying the groundwork for future distribution across multiple devices.

---

## Process Overview

For every generation, DENSE will:

1. **Create a Shard of the Last Dense Hidden Layer**:  
   At each generation, DENSE saves a shard representing the latest dense hidden layer. This snapshot captures the evaluation point in time, reducing computational costs and providing a starting point for future distributed processing.

2. **Generate 'LearnedOrNot' Files**:  
   These files track which evaluation shards have already been learned and which ones remain to be explored. This helps the system incrementally focus on learning new information without retraining the entire model. By analyzing the 'LearnedOrNot' files, the system identifies which layers need further exploration and refinement.

3. **Layer Sharding with Incremental Learning**:  
   DENSE uses a technique that can be described as **"Layer Sharding with Incremental Learning."** In each iteration, it loads the evaluation shard and explores potential new layers that could be added to the model. This allows the model to evolve by improving specific layers while retaining the rest of the network's learned structure. This process helps ensure that important information is preserved and the overall architecture remains stable as the network grows.

4. **Shard-to-Shard Linking and Model Distribution** (Future Step):  
   Once the model has been optimized on a single machine, the next phase will involve linking the shards together, enabling them to be distributed across multiple devices. This will be a key area of experimentation, but the core functionality is first being built and tested in a single-machine setup. The goal is to seamlessly distribute the network across devices without sacrificing the integrity of the model or its learned information.

---

## Approach Summary

This approach can be described as **Layer Sharding with Incremental Learning**. It focuses on optimizing the model layer by layer, saving computational resources by sharding the network, and setting the stage for eventual distribution. By exploring how layers can be added and refined without losing the structure of the model, DENSE aims to continually improve the network over generations while minimizing redundant computations.


![DENSE](./reminderdiagram.png)