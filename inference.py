import os
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from train import Net 
from PIL import Image
import numpy as np
from pathlib import Path
import argparse 
import pandas as pd 
from pathlib import Path 

data_csv_path = Path(__file__).parent / "data" / "cards.csv" 


if d:=os.getenv("DEVICE"):
  dev = d.lower()
  if dev not in ["cuda","cpu","mps"]: raise ValueError("Invalid device.")
  device = dev
else:
  device = "mps" if torch.backends.mps.is_available() else "cpu"



def get_labels():  
    itol = {}
    df = pd.read_csv(data_csv_path) 
    for idx, row in df.iterrows(): 
        if row["class index"] in itol:continue 
        itol[row["class index"]] = row["labels"]
    return itol

def prepare_image(path:str):   
    img = Image.open(path)  
    np_arr = np.array([img])
    x = torch.from_numpy(np_arr).float()  
    x = x.permute(0,3,1,2).to(device)
    dataset_mean = torch.mean(x, dim=(0, 2, 3))
    dataset_std = torch.std(x, dim=(0, 2, 3))
    normalize = transforms.Normalize(dataset_mean,dataset_std)
    X = normalize(x) 
    return X

if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("image", help="Path of image to classify") 

    args = parser.parse_args()

    DEFAULT_MODEL_PATH = "./card_net.pth" 
    model_path = os.getenv("MODEL",None) 
    if not model_path: 
        model_path = DEFAULT_MODEL_PATH 

    if not os.path.exists(model_path): 
        raise FileNotFoundError(f"{model_path} does not exist. Run train.py first.") 

    itol = get_labels()

    loaded_model = Net().to(device)
    loaded_model.load_state_dict(torch.load(model_path)) 
    loaded_model.eval()   
    x = prepare_image(args.image)
    with torch.no_grad(): 
        logits = loaded_model(x)   
        m = nn.Softmax(dim=1) 
        probs = m(logits)
        classification = torch.max(probs,dim=1)
        label = itol.get(int(classification.indices))
        print(label)

