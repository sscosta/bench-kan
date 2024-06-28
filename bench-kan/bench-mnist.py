import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os


from spline_kan import KAN
from rbf_kan import RBFKAN
from rational_quadratic_kan import RationalQuadraticKAN
from matern_kan import MaternKAN
from laplacian_kan import LaplacianKAN
from chebyshev_kan import ChebyshevKANMNIST, ChebyshevKAN
from cauchy_kan import CauchyKANMNIST, CauchyKAN

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

models =[
  KAN([28 * 28, 64, 10]),
  RBFKAN([28 * 28, 64, 10]),
  ChebyshevKANMNIST([28 * 28, 64, 10]),
  CauchyKANMNIST(),
  LaplacianKAN([28 * 28, 64, 10]),
  RationalQuadraticKAN([28 * 28, 64, 10],alpha=0.5),
  MaternKAN([28 * 28, 64, 10])
]
dataset_name = "MNIST"
learning_rate = 1e-3
for kan_model in models:
    print(sum(p.numel() for p in kan_model.parameters()))
    # Define model
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kan_model.to(device)

    # Set up the optimizer and loss function
    if isinstance(kan_model, LaplacianKAN):
        optimizer = torch.optim.SGD(kan_model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(kan_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    save_dir = f"/home/froot/kan_results/{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    save_file = f"{save_dir}{kan_model.__class__.__name__}_results.csv"
    f = open(save_file, "w")
    f.write("algorithm,epoch,train_loss,test_loss,test_accuracy,precision,recall,f1_score\n")

    # Create file object for writing results
    with open("results.txt", "w") as f:
        for epoch in range(10):
            # Train
            kan_model.train()
            train_loss = 0
            with tqdm(trainloader) as pbar:
                for i, (images, labels) in enumerate(pbar):
                    images = images.view(-1, 28 * 28).to(device)
                    optimizer.zero_grad()
                    output = kan_model(images)
                    loss = criterion(output, labels.to(device))
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
                for images, labels in valloader:
                    images = images.view(-1, 28 * 28).to(device)
                    output = kan_model(images)
                    val_loss += criterion(output, labels.to(device)).item()
                    val_accuracy += (
                        (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                    )
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(output.argmax(dim=1).cpu().numpy())
            val_loss /= len(valloader)
            val_accuracy /= len(valloader)

            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
            with open(save_file, "a") as f:
                f.write(f"{kan_model.__class__.__name__},{epoch+1},{train_loss},{val_loss},{val_accuracy},{precision},{recall},{f1}\n")

            print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

            # Update learning rate
            scheduler.step()