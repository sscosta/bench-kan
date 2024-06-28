import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

directory = sys.argv[1]

directory_parts = directory.split('/')
dataset_name = directory_parts[len(directory_parts) - 2]

output_directory = os.path.join(directory, dataset_name + '_plots')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

data_frames = []

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        
        df['filename'] = filename
        
        data_frames.append(df)

metrics = ['train_loss', 'test_loss', 'test_accuracy', 'precision', 'recall', 'f1_score']

for metric in metrics:
    plt.figure(figsize=(10, 6))

    handles, labels = [], []

    for df in data_frames:
        handle, = plt.plot(df['epoch'], df[metric], label=df['algorithm'].iloc[0])
        handles.append(handle)
        labels.append(df['algorithm'].iloc[0])
    
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Epoch')

    plt.ylim(0, 1)
    
    plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(data_frames))
    

    csv_filename = os.path.splitext(df['filename'].iloc[0])[0] 
    plot_filename = os.path.join(output_directory, f'{csv_filename}_{metric}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

print(f"Plots saved in {output_directory}")
