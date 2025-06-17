import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import yaml
from utils.video_utils import decode_video_frames

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None, max_frames=160):
        self.video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = decode_video_frames(video_path, max_frames=self.max_frames)
        # TODO: 特征提取（如光流、人脸、表情等）
        # features = extract_features(frames)
        # return frames, features
        return frames

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.video_dir = self.cfg['video_dir']
        self.batch_size = self.cfg['batch_size']
        self.num_workers = self.cfg['num_workers']
        self.max_frames = 160  # 可根据实际情况调整

    def setup(self, stage=None):
        self.dataset = VideoDataset(self.video_dir, max_frames=self.max_frames)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader() 