#!/Users/barshanitman/Desktop/Repos/ImageCard/env/bin/python 

import os
import torch
import torch.nn as nn
from load_dataset import get_training_dataset  
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

if d:=os.getenv("DEVICE"):
  dev = d.lower()
  if dev not in ["cuda","cpu","mps"]: raise ValueError("Invalid device.")
  device = dev
else:
  device = "mps" if torch.backends.mps.is_available() else "cpu" 


EPOCHS = os.getenv("EPOCHS",None) 
if not EPOCHS: 
    EPOCHS = 50 
else:
    EPOCHS = int(EPOCHS)

class Net(nn.Module):
	def __init__(self):
			super().__init__()
			self.conv1 = nn.Conv2d(3, 6, 5)
			self.pool = nn.MaxPool2d(2, 2)
			self.conv2 = nn.Conv2d(6, 16, 5)
			self.fc1 = nn.Linear(44944, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 53)

	def forward(self, x):
			x = self.pool(F.relu(self.conv1(x)))
			x = self.pool(F.relu(self.conv2(x)))
			x = torch.flatten(x, 1) # flatten all dimensions except batch
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x


if __name__ == "__main__":
    X,y = get_training_dataset() 
    X = X.permute(0,3,1,2).to(device) # Make it into N,C,H,W  
    y = y.to(device)
    dataset_mean = torch.mean(X, dim=(0, 2, 3))
    dataset_std = torch.std(X, dim=(0, 2, 3))
    normalize = transforms.Normalize(dataset_mean,dataset_std)  
    X = normalize(X) 

    train_dataset = TensorDataset(X, y)

    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True) # shuffle=True is crucial for training

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 

# Training Loop 
    print_every = 100 

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        # Use enumerate to get a batch index (i)
        for i, (X_batch, y_batch) in enumerate(train_loader):
            
            # Training steps
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Print statistics every `print_every` batches
            if (i + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}/{len(train_loader)}, Loss: {avg_loss:.3f}')
                
                # Reset running_loss after printing
                running_loss = 0.0

    print('Finished Training') 
    print('Saving model..')  
    save_path = "./card_net.pth"
    torch.save(net.state_dict(),save_path) 
    print('Model saved!')  
