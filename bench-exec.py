import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import math
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import torchvision

from spline_kan import KAN
from rbf_kan import RBFKAN
from rational_quadratic_kan import RationalQuadraticKAN
from matern_kan import MaternKAN
from laplacian_kan import LaplacianKAN
from chebyshev_kan import ChebyshevKANMNIST, ChebyshevKAN
from cauchy_kan import CauchyKANMNIST, CauchyKAN


directory = "/home/froot/preprocessed_data/"
dataset_names = ["monks", "credit","kc2","aids","haramb","har"]
label_columns = {"2024_02_21":"label","monks": "attr6", "kc2": "Class", "credit": "class", "iris": "species","mushroom":"class","mobile_price":"price_range", "MNIST":"label","aids":"target","haramb":"561","har":"Activity"}
map_num_classes = {"2024_02_21": 2,"monks": 2, "kc2": 2, "credit": 2, "iris": 3,"mushroom":2,"mobile_price":4, "MNIST":10,"aids":2,"haramb":7, "har":7}

num_epochs = 50
batch_size = 64
learning_rate = 0.01

def encode_string_columns(train_values, test_values):
    le = LabelEncoder()
    for col in train_values.select_dtypes(include=['object']).columns:
        train_values[col] = le.fit_transform(train_values[col])
        test_values[col] = le.transform(test_values[col])
    return train_values, test_values

for dataset_name in dataset_names:
    print(f"################# Dataset : {dataset_name} #################")
    train_filename = directory + dataset_name + "_train.csv"
    test_filename = directory + dataset_name + "_test.csv"

    num_classes = map_num_classes[dataset_name]

    train_values = pd.read_csv(train_filename)
    test_values = pd.read_csv(test_filename)
    # Encode string columns
    train_values, test_values = encode_string_columns(train_values, test_values)
      
    label_col = label_columns[dataset_name]
    # Split features and labels
    X_train = train_values.drop(columns=[label_col]).values
    y_train = train_values[label_col].values
    X_test = test_values.drop(columns=[label_col]).values
    y_test = test_values[label_col].values
   
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
   
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    n_features = X_train.shape[1]

    kan_models = [
      KAN([n_features, batch_size, num_classes]),
      RBFKAN([n_features, batch_size, num_classes]),
      LaplacianKAN([n_features, batch_size, num_classes]),
      ChebyshevKAN(n_features),
      CauchyKAN(n_features),
      RationalQuadraticKAN([n_features, batch_size, num_classes]),
      MaternKAN([n_features, batch_size, num_classes])
    ]
    for kan_model in kan_models:
        device = "cpu"
        kan_model.to(device)
        
        # Set up the optimizer and loss function
        if isinstance(kan_model, LaplacianKAN):
            optimizer = torch.optim.SGD(kan_model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(kan_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
     
        save_dir = f"/home/froot/kan_results/{dataset_name}/"
        os.makedirs(save_dir, exist_ok=True)
        save_file = f"{save_dir}{kan_model.__class__.__name__}_results.csv"
        with open(save_file, "w") as f:
            f.write("algorithm,epoch,train_loss,test_loss,test_accuracy,precision,recall,f1_score\n")
        
        for epoch in range(num_epochs):
            # Train
            kan_model.train()
            train_loss = 0
            with tqdm(trainloader) as pbar:
                for i, (features, labels) in enumerate(pbar):
                    features = features.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    output = kan_model(features)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    accuracy = (output.argmax(dim=1) == labels).float().mean()
                    pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
            train_loss /= len(trainloader)
            
            # Validation
            kan_model.eval()
            val_loss = 0
            val_accuracy = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for features, labels in testloader:
                    features = features.to(device)
                    labels = labels.to(device)
                    output = kan_model(features)
                    val_loss += criterion(output, labels).item()
                    val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(output.argmax(dim=1).cpu().numpy())
            
            val_loss /= len(testloader)
            val_accuracy /= len(testloader)
        
            # Compute precision, recall, and F1-score
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
            with open(save_file, "a") as f:
              f.write(f"{kan_model.__class__.__name__},{epoch+1},{train_loss},{val_loss},{val_accuracy},{precision},{recall},{f1}\n")
            
            print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")