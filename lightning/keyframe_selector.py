import numpy as np
import torch

def compute_centrality(features, adj):
    # features: [N, C], adj: [N, N]
    cos_sim = torch.matmul(features, features.T) / (
        features.norm(dim=1, keepdim=True) * features.norm(dim=1, keepdim=True).T + 1e-8)
    centrality = (adj * cos_sim).sum(dim=1)
    return centrality

def compute_distinctiveness(features):
    N = features.shape[0]
    cos_sim = torch.matmul(features, features.T) / (
        features.norm(dim=1, keepdim=True) * features.norm(dim=1, keepdim=True).T + 1e-8)
    distinctiveness = 1 - (cos_sim.sum(dim=1) - 1) / (N - 1)
    return distinctiveness

def compute_representativeness(features, selected):
    # selected: 已选关键帧索引集合
    N = features.shape[0]
    cos_sim = torch.matmul(features, features.T) / (
        features.norm(dim=1, keepdim=True) * features.norm(dim=1, keepdim=True).T + 1e-8)
    rep = []
    for i in range(N):
        if len(selected) == 0:
            rep.append(0)
        else:
            sims = cos_sim[i, selected]
            sims_expanded = sims.unsqueeze(1) # shape: [len(selected), 1]
            rep.append(torch.norm(sims_expanded * features[selected], p=2) / (torch.norm(sims, p=2) + 1e-8))
    return torch.tensor(rep, device=features.device)

def keyframe_selection(features, adj, k, alpha=0.4, beta=0.3, gamma=0.3, tau=0.8):
    # features: [N, C], adj: [N, N]
    N = features.shape[0]
    device = features.device
    selected = []
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    # 1. 计算三种分数
    centrality = compute_centrality(features, adj)
    distinctiveness = compute_distinctiveness(features)
    representativeness = compute_representativeness(features, selected)
    # 2. 综合得分
    score = alpha * centrality + beta * distinctiveness + gamma * representativeness
    # 3. 贪心选择
    for _ in range(k):
        idx = torch.argmax(score * (~mask)).item()
        selected.append(idx)
        mask[idx] = True
        # 更新代表性分数
        representativeness = compute_representativeness(features, selected)
        score = alpha * centrality + beta * distinctiveness + gamma * representativeness
        # 覆盖率终止
        cos_sim = torch.matmul(features, features[selected].T) / (
            features.norm(dim=1, keepdim=True) * features[selected].norm(dim=1, keepdim=True).T + 1e-8)
        coverage = cos_sim.max(dim=1)[0].mean().item()
        # if coverage >= tau:
        #     break
    selected = sorted(selected)
    return selected 