# 👁️ CivicEye: Autonomous Edge Surveillance System

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLO26-yellow)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Edge%2FWeb-ff69b4)

**CivicEye** is an enterprise-grade, autonomous computer vision pipeline designed to detect public offenses (such as littering) in real-time. Built for constrained edge environments, it features a specialized Python-based vision engine that natively triggers event logging and analytics into a full-stack dashboard.

---

## 🏗️ System Architecture

CivicEye bridges the gap between raw edge inference and actionable system-of-record logging through a **Hybrid AI Architecture**:

1. **The Vision Engine (Edge):** Processes 1080p CCTV feeds locally on Intel i7 CPUs. Instead of relying on cloud inference, the system uses compiled `.pt` weights trained on Kaggle Cloud GPUs, ensuring low latency and high privacy.
2. **The Intelligence Layer (Agentic Triggering):** Upon detecting a high-confidence offense, the vision engine acts as an autonomous agent, formatting the detection data (timestamps, bounding boxes, event types) and pushing it to the backend.
3. **The System of Record (MERN Dashboard):** A centralized command center that ingests the real-time API payloads, allowing administrators to review flagged events, monitor camera health, and analyze trends.

---

## ⚙️ Core Technical Features

### 🔍 Slicing Aided Hyper Inference (SAHI)
Detecting small objects (like discarded wrappers or bottles) in high-resolution 1080p feeds traditionally leads to pixel-blur degradation and feature loss during standard model downscaling. CivicEye implements **SAHI** to dynamically slice the inference grid, maintaining bounding box integrity and drastically improving small-object recall without requiring massive compute overhead.

### 🧠 Advanced Model Training & Dataset Curation
The core detection model utilizes **YOLO26**, optimized for rapid edge inference. To combat model hallucinations in "forced-choice" background environments, the dataset was curated using **Negative Example Injection**. By purposefully introducing 15% null images (backgrounds with no targets) into the training pipeline, the model's false-positive rate is severely reduced.

---

## 📊 Performance & Results

*(Note: The following metrics showcase the model's performance on the validation set and real-world edge testing.)*

### Inference Benchmarks
* **Hardware:** Local Intel i7 CPU
* **Resolution:** 1080p (Sliced via SAHI)
* **Average Inference Time:** `[Add ms here]` ms / frame
* **mAP@50:** `[Add % here]`%

### Precision vs. Recall
![Precision-Recall Curve](path/to/your/pr_curve.png)
> *Figure 1: Precision-Recall curve demonstrating the impact of Negative Example Injection on reducing false positives.*

### Real-World Detection Samples
| Base YOLO26 (Downscaled) | CivicEye (YOLO26 + SAHI) |
| :---: | :---: |
| ![Standard Inference](path/to/standard_inference.jpg) | ![SAHI Inference](path/to/sahi_inference.jpg) |
| *Small objects lost in downscaling.* | *Successfully isolated and classified litter.* |

### Training Loss Graphs
![Training Loss](path/to/your/loss_graph.png)
> *Figure 2: Bounding box and classification loss over `[X]` epochs.*

---

## 💻 Tech Stack

**Edge AI Pipeline:**
* Python
* YOLO26
* SAHI
* OpenCV / NumPy / Pandas

**Full-Stack Dashboard:**
* MongoDB
* Express.js
* React.js
* Node.js
* REST APIs

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Meet2909/CivicEye.git](https://github.com/Meet2909/CivicEye.git)
cd CivicEye

---

# 2. Setup the Vision Environment
```bash
python -m venv edge_env
source edge_env/bin/activate  # On Windows use `edge_env\Scripts\activate`
pip install -r requirements.txt
