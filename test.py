import os
import torch
import yaml
import numpy as np
from lightning.datamodule import VideoDataModule
from lightning.model import STGCN
from lightning.keyframe_selector import keyframe_selection
from lightning.utils import build_adjacency_matrix
from utils.video_utils import decode_video_frames, extract_features
import cv2

# 1. 加载配置
config_path = 'config/config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

video_dir = cfg['video_dir']
output_dir = cfg['output_dir']
os.makedirs(output_dir, exist_ok=True)

device = torch.device(cfg.get('device', 'cpu'))

# 2. 初始化模型
model = STGCN(config_path)
model.eval()
model.to(device)

# 3. 遍历视频
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
for vid in video_files:
    video_path = os.path.join(video_dir, vid)
    frames = decode_video_frames(video_path, max_frames=160)  # [T, H, W, C]
    N = frames.shape[0]
    if N == 0:
        continue
    # 4. 全特征提取
    features_np = extract_features(frames)  # [N, C]
    features = torch.tensor(features_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, C]
    # 5. 构建邻接矩阵
    adj = build_adjacency_matrix(features[0], neighbor_k=[1,2,4,8,16], sigma=1.0)
    adj = adj.unsqueeze(0)  # [1, N, N]
    # 6. ST-GCN前向
    with torch.no_grad():
        node_feats = model(features, adj)[0]  # [N, C]
    # 7. 关键帧选择
    selected = keyframe_selection(node_feats, adj[0], k=cfg['num_keyframes'],
                                 alpha=cfg['centrality_weight'],
                                 beta=cfg['distinctiveness_weight'],
                                 gamma=cfg['representativeness_weight'],
                                 tau=cfg['coverage_threshold'])
    # 8. 可视化与保存
    save_dir = os.path.join(output_dir, os.path.splitext(vid)[0])
    os.makedirs(save_dir, exist_ok=True)
    for idx in selected:
        frame = frames[idx]
        save_path = os.path.join(save_dir, f'keyframe_{idx:03d}.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"{vid} 关键帧索引: {selected}")
