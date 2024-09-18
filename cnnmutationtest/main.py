import os
import json
import matplotlib.pyplot as plt

# Define the path to the generations folder
generations_dir = "./host/generations"

# Function to load models from a generation folder and extract accuracy
def load_models_accuracies(generation_dir):
    accuracies = []
    for file_name in os.listdir(generation_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(generation_dir, file_name)
            with open(file_path, 'r') as f:
                model_data = json.load(f)
                # Handle missing 'LastTestAccuracy' key by setting accuracy to 0.0 or another default value
                accuracy = model_data.get("metadata", {}).get("lastTestAccuracy", 0.0)
                accuracies.append(accuracy)
    return sorted(accuracies, reverse=True)[:20]  # Top 20 models

# Function to collect accuracies from all generations
def collect_all_accuracies(generations_dir):
    generation_accuracies = []
    for generation_num in sorted(os.listdir(generations_dir), key=int):
        generation_path = os.path.join(generations_dir, generation_num)
        if os.path.isdir(generation_path):
            print(f"Processing generation {generation_num}...")
            accuracies = load_models_accuracies(generation_path)
            generation_accuracies.append((int(generation_num), accuracies))
            print(f"Finished processing generation {generation_num}.")
    return generation_accuracies

# Collect accuracies from all generations
generation_accuracies = collect_all_accuracies(generations_dir)

# Plot the accuracies for each generation
plt.figure(figsize=(10, 6))
for gen, accuracies in generation_accuracies:
    plt.plot([gen] * len(accuracies), accuracies, 'o', label=f'Generation {gen}' if gen % 10 == 0 else "", alpha=0.7)

plt.title('Top 20 Model Accuracies Per Generation')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='small')

# Save the plot as a JPEG image
output_path = './generation_accuracy_graph.jpg'
plt.savefig(output_path, format='jpg', bbox_inches='tight')

print(f"Graph saved to {output_path}")
