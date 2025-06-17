import torch
import numpy as np

def build_adjacency_matrix(features, neighbor_k=[1,2,4,8,16], sigma=1.0):
    """
    构建时空图邻接矩阵，支持多尺度邻接（相邻+跳跃），边权为高斯核或1/0。
    features: [N, C]
    neighbor_k: 跳跃步长集合
    sigma: 高斯核带宽
    返回: [N, N] 邻接矩阵
    """
    N = features.shape[0]
    adj = torch.zeros((N, N), device=features.device)
    for k in neighbor_k:
        for i in range(N):
            j = i + k
            if j < N:
                # 高斯核边权
                dist = torch.norm(features[i] - features[j])
                w = torch.exp(-dist**2 / (2 * sigma**2))
                adj[i, j] = w
                adj[j, i] = w
    # 自环
    adj += torch.eye(N, device=features.device)
    return adj 