# 🎬 ST-GCN4KeyFrames

A deep learning approach for video keyframe extraction using Spatial-Temporal Graph Convolutional Networks (ST-GCN).

## 📋 Overview

This project implements an innovative method for automatic video keyframe selection by leveraging ST-GCN to model temporal relationships between video frames. The approach combines multiple feature types including optical flow, scene changes, facial expressions, and deep visual features to create a comprehensive representation of video content.

## ✨ Features

- **🎯 Multi-modal Feature Extraction**: Combines optical flow, scene change detection, facial geometry, and deep visual features
- **🕸️ Graph-based Modeling**: Uses ST-GCN to capture temporal dependencies between frames
- **🧠 Intelligent Keyframe Selection**: Implements centrality, distinctiveness, and representativeness-based selection criteria
- **⚙️ Configurable Parameters**: Easy-to-modify configuration for different use cases
- **⚡ PyTorch Lightning Integration**: Clean, modular code structure with Lightning framework

## 🏗️ Architecture

### 🔍 Feature Extraction Pipeline

1. **⏱️ Temporal Features**: Frame indices and optical flow analysis
2. **🎨 Scene Analysis**: RGB histogram differences and SSIM-based scene change detection
3. **👤 Facial Features**: Face geometry, expressions, and pose estimation using MediaPipe
4. **🤖 Deep Features**: Pre-trained CNN features with PCA dimensionality reduction

### 🕸️ ST-GCN Model

- **🔗 Graph Construction**: K-nearest neighbor adjacency matrix with temporal constraints
- **🧮 Graph Convolution**: Multi-layer GCN with residual connections
- **🔀 Feature Fusion**: Combines all extracted features into unified representations

### 🎯 Keyframe Selection Algorithm

The selection process considers three key factors:

- **⭐ Centrality**: How well a frame represents the overall video content
- **💎 Distinctiveness**: How unique a frame is compared to others
- **🎯 Representativeness**: How well selected frames cover the entire video

## 📊 Keyframe Visualization

![Keyframe Extraction Process](assets/keyframes.png)

## 🚀 Installation

1. **📥 Clone the repository**:

```bash
git clone <repository-url>
cd ST-GCN4KeyFrames
```

2. **🐍 Create a virtual environment**:

```bash
python -m venv stgcn-keyframes-env
source stgcn-keyframes-env/bin/activate  # On Windows: stgcn-keyframes-env\Scripts\activate
```

3. **📦 Install dependencies**:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### ⚙️ Configuration

Edit `config/config.yaml` to customize parameters:

```yaml
# Basic settings
video_dir: "./data/videos"      # Input video directory
output_dir: "./data/outputs"    # Output directory for keyframes
batch_size: 4
num_workers: 4
num_keyframes: 16              # Number of keyframes to extract

# ST-GCN model parameters
in_channels: 136               # Input feature dimension
hidden_channels: [256, 128, 64] # Hidden layer dimensions
num_layers: 3                  # Number of GCN layers

# Keyframe selection weights
centrality_weight: 0.4         # Weight for centrality score
distinctiveness_weight: 0.3    # Weight for distinctiveness score
representativeness_weight: 0.3 # Weight for representativeness score
coverage_threshold: 0.8        # Coverage threshold for early stopping

# Device configuration
device: "cuda:0"              # Use "cpu" for CPU-only processing
```

### 🎬 Running Keyframe Extraction

1. **📁 Place your videos** in the `data/videos` directory
2. **▶️ Run the extraction script**:

```bash
python test.py
```

The script will:

- 🔄 Process all `.mp4` files in the input directory
- 🔍 Extract comprehensive features from each frame
- 🕸️ Apply ST-GCN to model temporal relationships
- 🎯 Select optimal keyframes based on the configured criteria
- 💾 Save selected keyframes as JPEG images in the output directory

### 📂 Output Structure

```
data/outputs/
├── video1/
│   ├── keyframe_000.jpg
│   ├── keyframe_015.jpg
│   └── ...
├── video2/
│   ├── keyframe_000.jpg
│   ├── keyframe_012.jpg
│   └── ...
```

## 📁 Project Structure

```
ST-GCN4KeyFrames/
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   ├── videos/              # Input video directory
│   └── outputs/             # Output keyframes directory
├── lightning/
│   ├── datamodule.py        # PyTorch Lightning data module
│   ├── keyframe_selector.py # Keyframe selection algorithm
│   ├── model.py             # ST-GCN model implementation
│   └── utils.py             # Utility functions
├── utils/
│   └── video_utils.py       # Video processing and feature extraction
├── test.py                  # Main execution script
├── requirements.txt         # Python dependencies
└── readme.md               # This file
```

## 📦 Dependencies

- **⚡ PyTorch Lightning**: Deep learning framework
- **🔥 PyTorch**: Neural network library
- **📹 OpenCV**: Video processing
- **👤 MediaPipe**: Face detection and landmark extraction
- **🔬 scikit-learn**: Machine learning utilities
- **🖼️ scikit-image**: Image processing
- **🔢 NumPy**: Numerical computing
- **📊 Matplotlib**: Visualization

## 🔧 Key Components

### 🔍 Feature Extraction (`utils/video_utils.py`)

- **🌊 Optical Flow**: Motion analysis between consecutive frames
- **🎨 Scene Change Detection**: RGB histogram differences and SSIM
- **👤 Facial Analysis**: Face geometry, expressions, and pose estimation
- **🤖 Deep Features**: Pre-trained CNN features with PCA

### 🕸️ ST-GCN Model (`lightning/model.py`)

- **🔗 GraphConvBlock**: Residual graph convolution with batch normalization
- **🧠 STGCN**: Main model combining multiple GCN layers

### 🎯 Keyframe Selection (`lightning/keyframe_selector.py`)

- **⭐ Centrality Computation**: Frame importance based on graph structure
- **💎 Distinctiveness**: Uniqueness measurement
- **🎯 Representativeness**: Coverage optimization
- **🔄 Greedy Selection**: Iterative frame selection algorithm

## ⚡ Performance Considerations

- **🚀 GPU Acceleration**: Configure `device: "cuda:0"` for GPU processing
- **💾 Memory Usage**: Adjust `batch_size` and `max_frames` based on available memory
- **⚡ Processing Speed**: Reduce `num_keyframes` for faster processing
- **🎯 Feature Quality**: Modify feature extraction parameters in `video_utils.py`

## 🛠️ Customization

### ➕ Adding New Features

1. **🔧 Implement feature extraction function** in `utils/video_utils.py`
2. **🔄 Update `extract_features()`** to include new features
3. **📊 Adjust `in_channels`** in configuration

### 🎯 Modifying Selection Criteria

1. **⚖️ Edit weights** in `config.yaml`
2. **🧮 Modify selection algorithm** in `lightning/keyframe_selector.py`
3. **➕ Add new scoring functions** as needed

### 🏗️ Model Architecture Changes

1. **🔧 Modify `lightning/model.py`** for different GCN architectures
2. **📊 Adjust `hidden_channels` and `num_layers`** in configuration

## 📄 License

[Add your license information here]

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

[Add your contact information here]
