import os
import torch
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import cpu_count

class FI2010_Orderbook_Optimized(Dataset):
    def __init__(
        self,
        root_path,
        window_size=100,
        overlap_percent=0,
        num_workers=cpu_count() // 2,
        verbose=True,
    ):
        self.window_size = window_size
        self.overlap_percent = overlap_percent
        self.verbose = verbose
        self.num_workers = num_workers

        self.encoder = OneHotEncoder(sparse_output=False)

        if not (0 <= overlap_percent < 100):
            raise ValueError("overlap_percent must be between 0 and 100.")
        self.step_size = max(1, int(window_size * (1 - overlap_percent / 100)))

        self.data_path = root_path

        self.data, self.labels, self.file_mapping = self._load_data()

    def _load_data(self):
        csv_files = [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.endswith(".csv")
        ]
        all_windows = []
        raw_price_movements = []
        file_mapping = []

        progress_bar = tqdm(total=len(csv_files), desc="Processing CSV Files")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._process_file, file): file for file in csv_files}
            for future in as_completed(futures):
                try:
                    windows, movements, file_name = future.result()
                    all_windows.extend(windows)
                    raw_price_movements.extend(movements)
                    file_mapping.extend([file_name] * len(windows))
                except Exception as e:
                    print(f"Error processing file {futures[future]}: {e}")
                finally:
                    progress_bar.update(1)

        progress_bar.close()

        raw_price_movements = np.array(raw_price_movements).reshape(-1, 1)
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

        return all_windows, labels, file_mapping

    def _process_file(self, file_path):
        data = pd.read_csv(file_path, header=None).to_numpy()
        features = data[0:40]
        labels = data[-5:]

        T = data.shape[1]
        windows = []
        price_movements = []

        for start in range(0, T - self.window_size + 1, self.step_size):
            windows.append(features[:, start:start + self.window_size])
            movement = labels[-1, start + self.window_size - 1]
            price_movements.append(movement)

        if (T - self.window_size) % self.step_size != 0:
            windows.append(features[:, -self.window_size:])
            price_movements.append(labels[-1, -1])

        return windows, price_movements, file_path

    def get_dataset_by_files(self, file_list):
        indices = [
            i for i, file_name in enumerate(self.file_mapping) if file_name in file_list
        ]
        subset_data = [self.data[i] for i in indices]
        subset_labels = [self.labels[i] for i in indices]
        return CustomSubsetDataset(subset_data, subset_labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)


class CustomSubsetDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)


# Main Method to Test
def main():
    root_path = "/home/jackklingenberg/Projects/LOBnet/data/pruned_auc_zscore_training"  
    window_size = 100
    overlap_percent = 25  # 25% overlap

    dataset = FI2010_Orderbook_Optimized(
        root_path=root_path,
        window_size=window_size,
        overlap_percent=overlap_percent,
        num_workers=9,
        verbose=True,
    )

    print(f"Dataset loaded with {len(dataset)} windows.")
    print(f"Sample window shape: {dataset[0][0].shape}")
    print(f"Sample label shape: {dataset[0][1].shape}")

    file_name = "Train_Dst_NoAuction_ZScore_CF_8.csv"
    subset = dataset.get_dataset_by_files([os.path.join(root_path, file_name)])

    print(f"Subset dataset loaded with {len(subset)} windows from {file_name}.")
    print(f"Sample window shape in subset: {subset[0][0].shape}")
    print(f"Sample label shape in subset: {subset[0][1].shape}")

if __name__ == "__main__":
    main()
