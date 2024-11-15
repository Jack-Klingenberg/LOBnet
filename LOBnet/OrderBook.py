import torch 
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class FI2010_Orderbook(Dataset):
    def __init__(self, file_path, venue="NoAuction", normalization_type="Zscore", split="Training"):
        self.root_path = file_path
        self.venue = venue
        self.normalization_type=normalization_type
        self.normalization_index = str( ["Zscore","MinMax","DecPre"].index(self.normalization_type)+1 )+"."
        self.split = split

        self.data_path = os.path.join(self.root_path, venue, self.normalization_index+venue+"_"+normalization_type, venue+"_"+normalization_type+"_"+split)

        self.file_event_counts = {}

        for file_name in os.listdir(self.data_path):
            if file_name.endswith('.txt'):
                path = os.path.join(self.data_path, file_name)
                
                with open(path, 'r') as file:
                    csv = pd.read_csv(path, delimiter="  ")
        
        print(self.file_event_counts)
            
            

    def __getitem__(self, idx):
        return idx



print("Start")
orderbook = FI2010_Orderbook("../data/published/BenchmarkDatasets/BenchmarkDatasets")
print("End")