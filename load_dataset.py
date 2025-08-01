#!/Users/barshanitman/Desktop/Repos/ImageCard/env/bin/python

import os
import torch  
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple

data_dir = str(Path(__file__).parent / "data" / "cards.csv")
base_dir = str(Path(__file__).parent  / "data") 

def get_training_dataset() -> Tuple:
  if d:=os.getenv("DEVICE"):  
    dev = d.lower()
    if dev not in ["cuda","cpu","mps"]: raise ValueError("Invalid device.")
    device = dev
  else:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

  df = pd.read_csv(data_dir)  

  # Train Data 
  df_train = df[df["data set"] == "train"]  

  imgs = [] 
  y = []

  for idx,row in df_train.iterrows(): 
    if "output" not in row["filepaths"]:
      img = Image.open(f'{base_dir}/{row["filepaths"]}') 
      np_arr = np.array(img)
      imgs.append(np_arr)  
      y.append(row["class index"])
  X_numpy = np.concatenate(imgs,axis=0)  
  X = torch.from_numpy(X_numpy) 
  y = torch.tensor(y)
  return X,y
