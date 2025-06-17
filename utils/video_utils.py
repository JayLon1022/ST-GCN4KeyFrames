import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import mediapipe as mp
import torch
import torchvision.transforms as T
import torchvision.models as models
from sklearn.decomposition import PCA

def decode_video_frames(video_path, max_frames=160):
    """
    解码视频，等间隔采样最多 max_frames 帧，返回 [T, H, W, C] numpy 数组。
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, min(max_frames, total_frames)).astype(int)
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    frames = np.stack(frames) if len(frames) > 0 else np.zeros((0, 224, 224, 3), dtype=np.uint8)
    return frames 

def extract_time_index_features(frames):
    """
    帧序列索引特征，归一化时间戳 [N, 1]
    """
    N = frames.shape[0]
    idx = np.arange(N).reshape(-1, 1)
    time_idx = idx / max(N-1, 1)
    return time_idx.astype(np.float32)

def extract_optical_flow_features(frames, num_bins=8):
    """
    光流特征：平均幅度、方向直方图、运动一致性等 [N, D]
    """
    N = frames.shape[0]
    H, W = frames.shape[1:3]
    flows = []
    for i in range(1, N):
        prev = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
    # 统计特征
    feats = []
    for i in range(N):
        if i == 0 or i > len(flows):
            # 首帧或越界补零
            feats.append(np.zeros(2 + num_bins + 1, dtype=np.float32))
        else:
            flow = flows[i-1]
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mean_mag = np.mean(mag)
            mean_ang = np.mean(ang)
            # 方向直方图
            hist, _ = np.histogram(ang, bins=num_bins, range=(0, 2*np.pi), density=True)
            # 运动一致性（方向方差）
            cons = np.var(ang)
            feat = np.concatenate([[mean_mag, mean_ang], hist, [cons]])
            feats.append(feat)
    feats = np.stack(feats, axis=0)  # [N, D]
    return feats

def extract_scene_change_features(frames, hist_bins=16):
    """
    场景变化检测特征：RGB直方图差分、SSIM [N, D]
    """
    N = frames.shape[0]
    feats = []
    for i in range(N):
        if i == 0:
            feats.append(np.zeros(hist_bins * 3 + 1, dtype=np.float32))
        else:
            # RGB直方图差分
            hist_diff = []
            for c in range(3):
                h1 = cv2.calcHist([frames[i-1]], [c], None, [hist_bins], [0, 256])
                h2 = cv2.calcHist([frames[i]], [c], None, [hist_bins], [0, 256])
                h1 = cv2.normalize(h1, h1).flatten()
                h2 = cv2.normalize(h2, h2).flatten()
                hist_diff.append(np.abs(h1 - h2))
            hist_diff = np.concatenate(hist_diff)
            # SSIM
            ssim_val = ssim(frames[i-1], frames[i], channel_axis=2, data_range=255)
            feat = np.concatenate([hist_diff, [ssim_val]])
            feats.append(feat)
    feats = np.stack(feats, axis=0)
    return feats

def extract_face_geometry_features(frames):
    """
    人脸几何特征：人脸框比例、三维姿态角 [N, D]
    """
    N = frames.shape[0]
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    feats = []
    for i in range(N):
        img = frames[i]
        h, w, _ = img.shape
        results = face_mesh.process(img)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            xs = [p.x for p in lm.landmark]
            ys = [p.y for p in lm.landmark]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            # 人脸框比例
            fw = (max_x - min_x)
            fh = (max_y - min_y)
            ratio = fw / (fh + 1e-6)
            # 取关键点估算三维姿态角（简化版）
            # 以左右眼、鼻尖、嘴角等
            left_eye = lm.landmark[33]
            right_eye = lm.landmark[263]
            nose = lm.landmark[1]
            mouth_left = lm.landmark[61]
            mouth_right = lm.landmark[291]
            # 俯仰角 pitch（y方向）
            pitch = np.arctan2(nose.y - (left_eye.y + right_eye.y)/2, nose.x - (left_eye.x + right_eye.x)/2)
            # 偏航角 yaw（x方向）
            yaw = np.arctan2(right_eye.x - left_eye.x, right_eye.y - left_eye.y)
            # 翻滚角 roll（嘴角）
            roll = np.arctan2(mouth_right.y - mouth_left.y, mouth_right.x - mouth_left.x)
            feat = np.array([fw, fh, ratio, pitch, yaw, roll], dtype=np.float32)
        else:
            feat = np.zeros(6, dtype=np.float32)
        feats.append(feat)
    feats = np.stack(feats, axis=0)
    face_mesh.close()
    return feats

def eye_aspect_ratio(eye_points):
    # 计算眼睛开合度（EAR）
    # 选用6个点，参考dlib/mediapipe关键点
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(mouth_points):
    # 计算嘴巴开合度（MAR）
    A = np.linalg.norm(mouth_points[2] - mouth_points[6])
    B = np.linalg.norm(mouth_points[3] - mouth_points[5])
    C = np.linalg.norm(mouth_points[0] - mouth_points[4])
    return (A + B) / (2.0 * C + 1e-6)

def extract_expression_features(frames):
    """
    面部表情特征：眼睛开合度、嘴巴开合度、眉毛高度等 [N, D]
    """
    N = frames.shape[0]
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    feats = []
    for i in range(N):
        img = frames[i]
        h, w, _ = img.shape
        results = face_mesh.process(img)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            # 取左右眼、嘴巴、眉毛关键点
            # mediapipe 468点，参考官方文档
            lm_np = np.array([[p.x * w, p.y * h] for p in lm.landmark])
            # 左眼(33, 160, 158, 133, 153, 144)
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            left_eye = lm_np[left_eye_idx]
            left_ear = eye_aspect_ratio(left_eye)
            # 右眼(263, 387, 385, 362, 380, 373)
            right_eye_idx = [263, 387, 385, 362, 380, 373]
            right_eye = lm_np[right_eye_idx]
            right_ear = eye_aspect_ratio(right_eye)
            # 嘴巴(78, 81, 13, 311, 308, 402, 14)
            mouth_idx = [78, 81, 13, 311, 308, 402, 14]
            mouth = lm_np[mouth_idx]
            mar = mouth_aspect_ratio(mouth)
            # 左眉毛(70, 63, 105, 66, 107)
            left_brow_idx = [70, 63, 105, 66, 107]
            left_brow = lm_np[left_brow_idx]
            left_brow_height = np.mean(left_brow[:,1])
            # 右眉毛(336, 296, 334, 293, 300)
            right_brow_idx = [336, 296, 334, 293, 300]
            right_brow = lm_np[right_brow_idx]
            right_brow_height = np.mean(right_brow[:,1])
            feat = np.array([left_ear, right_ear, mar, left_brow_height, right_brow_height], dtype=np.float32)
        else:
            feat = np.zeros(5, dtype=np.float32)
        feats.append(feat)
    feats = np.stack(feats, axis=0)
    face_mesh.close()
    return feats

def extract_deep_features(frames, pca_dim=64, device='cpu'):
    """
    深度层次特征：ResNet-50高维特征，PCA降维 [N, pca_dim]
    """
    N = frames.shape[0]
    # 预处理
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imgs = [transform(frames[i]) for i in range(N)]
    imgs = torch.stack(imgs, dim=0).to(device)
    # 加载ResNet-50
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 去除fc层
    resnet.eval()
    resnet.to(device)
    with torch.no_grad():
        feats = resnet(imgs).squeeze(-1).squeeze(-1).cpu().numpy()  # [N, 2048]
    # PCA降维
    if feats.shape[0] > 1:
        pca = PCA(n_components=pca_dim)
        feats_pca = pca.fit_transform(feats)
    else:
        feats_pca = np.zeros((N, pca_dim), dtype=np.float32)
    return feats_pca.astype(np.float32)

def extract_features(frames):
    """
    综合特征提取入口，返回[N, C]特征矩阵
    """
    feats = []
    # 1. 帧序列索引特征
    feats.append(extract_time_index_features(frames))
    # 2. 光流特征
    feats.append(extract_optical_flow_features(frames))
    # 3. 场景变化特征
    feats.append(extract_scene_change_features(frames))
    # 4. 人脸几何特征
    feats.append(extract_face_geometry_features(frames))
    # 5. 表情特征
    feats.append(extract_expression_features(frames))
    # 6. 深度特征
    feats.append(extract_deep_features(frames, pca_dim=64, device='cpu'))
    # 拼接
    feats = [f for f in feats if f is not None]
    features = np.concatenate(feats, axis=1)
    return features 