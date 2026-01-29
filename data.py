
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class GHTDataset(Dataset):
    def __init__(self, root, size=512, train_final_state_model=True):
        self.size = size
        self.root = Path(root)
        
        # Select target directories based on model type
        if train_final_state_model:
            self.dir_tgt = self.root / 'i_state' # i_state and p_state for final state model
            self.dir_txt = self.root / 'p_state'
        else:
            self.dir_tgt = self.root / 'i_action' # i_action and p_action for action model
            self.dir_txt = self.root / 'p_action'

        self.dir_init = self.root / 'i_init'
        
        if not self.dir_init.exists() or not self.dir_tgt.exists():
            raise FileNotFoundError(f"ERROR - Data folders missing")

        exts = {'.jpg', '.jpeg', '.png'}
        self.files = sorted([f for f in self.dir_init.iterdir() if f.suffix.lower() in exts])
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        src_path = self.files[idx]

        tgt_path = self.dir_tgt / src_path.name
        txt_path = self.dir_txt / (src_path.stem + '.txt')

        src = Image.open(src_path).convert('RGB')
        tgt = Image.open(tgt_path).convert('RGB')
        
        with open(txt_path, 'r') as f:
            txt = f.read().strip()

        return {
            'src': self.transform(src),
            'tgt': self.transform(tgt),
            'txt': txt
        }