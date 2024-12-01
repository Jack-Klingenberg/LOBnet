import os
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import cpu_count

class FI2010_Orderbook_Optimized(Dataset):
    def __init__(
        self,
        root_path,
        venue="NoAuction",
        normalization_type="Zscore",
        split="Training",
        window_size=100,
        overlap_percent=0,  
        num_workers=cpu_count()//2, # set num processes to half the number of CPU cores
        verbose=True,
    ):
        self.root_path = root_path
        self.venue = venue
        self.normalization_type = normalization_type
        self.split = split
        self.window_size = window_size
        self.overlap_percent = overlap_percent
        self.verbose = verbose
        self.num_workers = num_workers

        self.encoder = OneHotEncoder(sparse_output=False)
        # Windowsize stuff
        if not (0 <= overlap_percent < 100):
            raise ValueError("overlap_percent must be between 0 and 100.")
        self.step_size = max(1, int(window_size * (1 - overlap_percent / 100))) 
        
        # File stuff 
        normalization_index = str(["Zscore", "MinMax", "DecPre"].index(self.normalization_type) + 1) + "."
        self.data_path = os.path.join(
            self.root_path,
            venue,
            normalization_index + venue + "_" + normalization_type,
            venue + "_" + normalization_type + "_" + split,
        )

        self.data, self.labels = self._load_data()

    def _load_data(self):
        csv_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(".csv")]
        all_windows = []
        raw_price_movements = []  # store raw movements first

        # first pass: collect all windows and raw price movements
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self._process_file, csv_files),
                    total=len(csv_files),
                    desc="Processing CSV Files",
                )
            )

        for windows, movements in results:
            all_windows.extend(windows)
            raw_price_movements.extend(movements)

        # then convert to numpy array and reshape for OneHotEncoder
        raw_price_movements = np.array(raw_price_movements).reshape(-1, 1)
        
        # fit and transform the price movements to get labels
        labels = self.encoder.fit_transform(raw_price_movements)
        
        if self.verbose:
            print("\nRaw price movement statistics:")
            print(f"Mean: {np.mean(raw_price_movements):.4f}")
            print(f"Std: {np.std(raw_price_movements):.4f}")
            print(f"Min: {np.min(raw_price_movements):.4f}")
            print(f"Max: {np.max(raw_price_movements):.4f}")
            
            label_indices = self.encoder.transform(raw_price_movements).argmax(axis=1)
            print("\nLabel distribution:")
            print(pd.Series(label_indices).value_counts().sort_index())

        return all_windows, labels

    def _process_file(self, file_path):
        if self.verbose:
            print(f"Processing file: {file_path}")

        # load features (first 40 rows)
        df_features = pd.read_csv(file_path, header=None, nrows=40)
        data = df_features.to_numpy()
        
        # load labels (last 5 rows)
        df_labels = pd.read_csv(file_path, header=None)
        labels = df_labels.iloc[-5:].to_numpy()

        T = data.shape[1]
        windows = []
        price_movements = []

        for start in range(0, T - self.window_size + 1, self.step_size):
            windows.append(data[:, start:start + self.window_size])
            movement = labels[-1, start + self.window_size - 1]
            price_movements.append(movement)

        if (T - self.window_size) % self.step_size != 0:
            windows.append(data[:, -self.window_size:])
            price_movements.append(labels[-1, -1])

        return windows, price_movements

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)  # one-hot encoded labels
        )
    
    def __len__(self):
        return len(self.data)

# Main Method to Test
# def main():
#     root_path = "../data/BenchmarkDatasets_csv"  
#     window_size = 100
#     overlap_percent = 25  # 25% overlap

#     dataset = FI2010_Orderbook_Optimized(
#         root_path=root_path,
#         venue="NoAuction",
#         normalization_type="Zscore",
#         split="Training",
#         window_size=window_size,
#         overlap_percent=overlap_percent,
#         num_workers=6,
#         verbose=True,
#     )

#     print(f"Dataset loaded with {len(dataset)} windows.")
#     print(f"Sample window shape: {dataset[0].shape}")


# if __name__ == "__main__":
#     main()
