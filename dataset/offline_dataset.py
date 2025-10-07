from torch.utils.data import Dataset, DataLoader
import torch
import glob
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def load_single_file(file_path):
    return torch.load(file_path, map_location='cpu')


class OfflineDataset(Dataset):
    
    def __init__(self, args, max_workers=None, files_per_batch=100):
        data_dir = os.path.join(args.dataset_dir, args.dataset, 'offline_data')
        
        pattern = os.path.join(data_dir, "*.pt")
        self.all_file_paths = sorted(glob.glob(pattern))
        
        if not self.all_file_paths:
            raise ValueError(f"No .pt files found in {data_dir}")
        
        self.max_workers = max_workers
        self.files_per_batch = files_per_batch
        self.current_file_index = 0
        self.data = []
        print(f"Found {len(self.all_file_paths)} files in {data_dir}")
    
    def reload_next_batch(self):
        if self.current_file_index >= len(self.all_file_paths):
            self.current_file_index = 0
            random.shuffle(self.all_file_paths)
            print("Shuffled files for new epoch")
        
        end_idx = min(self.current_file_index + self.files_per_batch, len(self.all_file_paths))
        current_batch_paths = self.all_file_paths[self.current_file_index:end_idx]
        
        # print(f"Loading files {self.current_file_index+1}-{end_idx} ({len(current_batch_paths)} files)...")
        
        # Load files in parallel
        self.data = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(load_single_file, path): path for path in current_batch_paths}
            
            with tqdm(total=len(current_batch_paths), desc="Loading batch") as pbar:
                for future in as_completed(future_to_path):
                    data = future.result()
                    self.data.append(data)
                    pbar.update(1)
        
        self.current_file_index = end_idx
        # print(f"Successfully loaded {len(self.data)} batches")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    
    # Create dataset with small batch size for demo
    dataset = OfflineDataset(args, files_per_batch=5)
    dataset.reload_next_batch()
    dataset.reload_next_batch()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, batch in enumerate(dataloader):
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")