import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Directory where CSV files are located (provide as command-line argument)
directory = sys.argv[1]

# Extract dataset name from directory path
directory_parts = directory.split('/')
dataset_name = directory_parts[len(directory_parts) - 2]

# Create output directory based on current working directory
output_directory = os.path.join(directory, dataset_name + '_plots')

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List to store DataFrame objects for each file
data_frames = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Read CSV file into DataFrame
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        
        # Add filename as a column in the DataFrame for later use
        df['filename'] = filename
        
        # Append DataFrame to the list
        data_frames.append(df)

# Define metrics to plot
metrics = ['train_loss', 'test_loss', 'test_accuracy', 'precision', 'recall', 'f1_score']

# Plotting each metric
for metric in metrics:
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Variable to store legend handles and labels
    handles, labels = [], []

    for df in data_frames:
        # Plot each algorithm's metric
        handle, = plt.plot(df['epoch'], df[metric], label=df['algorithm'].iloc[0])
        handles.append(handle)
        labels.append(df['algorithm'].iloc[0])
    
    # Set plot labels and title
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Epoch')
    
    # Limit y-axis to 0-1
    plt.ylim(0, 1)
    
    # Place legend below the plot
    plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(data_frames))
    
    # Save the plot to a file named after the original CSV file
    csv_filename = os.path.splitext(df['filename'].iloc[0])[0]  # Use the first DataFrame's filename
    plot_filename = os.path.join(output_directory, f'{csv_filename}_{metric}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()  # Close the current figure to release memory

# Print the output directory where plots are saved
print(f"Plots saved in {output_directory}")
