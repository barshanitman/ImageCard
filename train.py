#!/Users/barshanitman/Desktop/Repos/ImageCard/env/bin/python 

import os
import torch
import torch.nn as nn
from load_dataset import get_training_dataset  
import torchvision.transforms as transforms
import torch.optim as optim

if d:=os.getenv("DEVICE"):
  dev = d.lower()
  if dev not in ["cuda","cpu","mps"]: raise ValueError("Invalid device.")
  device = dev
else:
  device = "mps" if torch.backends.mps.is_available() else "cpu"  

X,y = get_training_dataset() 
X = X.permute(0,3,1,2) # Make it into N,C,H,W  
X = transforms.Normalize(X) 
breakpoint()

class Net(nn.Module):
	def __init__(self):
			super().__init__()
			self.conv1 = nn.Conv2d(3, 6, 5)
			self.pool = nn.MaxPool2d(2, 2)
			self.conv2 = nn.Conv2d(6, 16, 5)
			self.fc1 = nn.Linear(16 * 5 * 5, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
			x = self.pool(F.relu(self.conv1(x)))
			x = self.pool(F.relu(self.conv2(x)))
			x = torch.flatten(x, 1) # flatten all dimensions except batch
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x


net = Net() 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


