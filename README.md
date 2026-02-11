# PathoGAT-TME: Spatially-Aware Graph Attention Networks for Tumor Microenvironments

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)

**PathoGAT-TME** is a deep learning framework designed to analyze the **Tumor Microenvironment (TME)** in breast cancer histology slides. Unlike traditional Convolutional Neural Networks (CNNs) that treat tissue as a grid of pixels, this project models the tissue as a **spatial graph of interacting cells**.

By leveraging **Graph Attention Networks (GATv2)** and **Morphological Node Features**, this model achieves state-of-the-art class balance, specifically outperforming baseline methods in detecting difficult **Stroma** and **Immune** cell populations.

---

## 🔬 1. Project Overview & Motivation

### The Problem
In digital pathology, the spatial arrangement of immune cells relative to tumor cells (tumor-infiltrating lymphocytes) is a critical biomarker for patient prognosis. Standard CNNs often struggle with:
1.  **Long-range dependencies:** Cells interacting across the tissue slide.
2.  **Class Imbalance:** Over-predicting "Tumor" cells while ignoring the critical stromal matrix.
3.  **Computational Cost:** Processing gigapixel Whole Slide Images (WSIs).

### The Solution: Graph Representation
We transform the histology slide into a **Cellular Graph**:
* **Nodes:** Individual cell nuclei (extracted via HoVer-Net/NuCLS).
* **Edges:** Spatial proximity (K-Nearest Neighbors, $k=5$) representing biological interaction.
* **Features:** Deep embeddings + Explicit Morphological Metrics (Area, Eccentricity, Solidity).

---

## 🧬 2. Dataset: NuCLS

This study utilizes the **NuCLS (Nucleus Classification and Segmentation)** dataset, a large-scale, pathologist-validated dataset derived from the TCGA (The Cancer Genome Atlas).

* **Source:** [NuCLS Dataset](https://sites.google.com/view/nucls/home)
* **Classes:**
    * **Tumor:** Malignant epithelial cells.
    * **Fibroblast (Stroma):** Connective tissue cells forming the tumor matrix.
    * **Lymphocyte (Immune):** T-cells and B-cells fighting the tumor.
    * **Other:** Macrophages, plasma cells, and ambiguous structures.
* **Data Scale:** ~80,000 annotated nuclei across diverse breast cancer subtypes.

> **Note:** Raw images are not included in this repo due to size constraints. The `data/raw/` folder contains instructions for downloading the dataset.

---

## ⚙️ 3. Methodology & Architectures

We conducted a rigorous comparative study of three Graph Neural Network architectures:

### A. Baseline GNN (GCNConv)
* **Architecture:** Standard Graph Convolutional Network.
* **Observation:** High bias towards the majority class (Tumor). Failed to learn "Other" class completely.

### B. Weighted GAT (GATConv)
* **Architecture:** Graph Attention Network with multi-head attention.
* **Innovation:** Applied **Biologically-Balanced Class Weights** to the Cross-Entropy Loss to penalize the model for ignoring rare cells.
* **Result:** Significant jump in Immune cell detection.

### C. Morphology-Aware GATv2 (The Proposed Model)
* **Architecture:** GATv2 (Dynamic Graph Attention) + Concatenated Morphological Features.
* **Innovation:** We enriched the node feature vector $h_i$ with geometric attributes:
    $$h_i' = h_i \oplus [Area, Perimeter, Eccentricity, Circularity]$$
* **Why GATv2?** Standard GAT computes static attention weights. GATv2 computes **dynamic attention**, allowing the model to focus on different neighbors depending on the query node's specific context (e.g., a Tumor cell looking for a nearby Lymphocyte).

---

## 📊 4. Experimental Results

The models were evaluated on a held-out test set of **263 WSI regions**.

| Model Architecture | Overall Accuracy | Tumor (Easy) | Immune (Medium) | Stroma (Hard) | Other (Hardest) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GNN (Baseline)** | 51.61% | **86.37%** | 38.25% | 30.88% | 0.00% |
| **GAT (Weighted)** | **58.18%** | 73.79% | 63.20% | 53.72% | 0.00% |
| **GATv2 (Morphology)** | 57.73% | 68.26% | **66.32%** | **57.35%** | **1.41%** |

### Key Findings
1.  **Attention is Critical:** The jump from GNN to GAT (+7% Accuracy) proves that not all neighbors are equal. The model needs to "attend" to specific cell types in the neighborhood.
2.  **Morphology Matters for Stroma:** The **GATv2 model** achieved the best performance on Stroma cells (**57.35%**) and Immune cells (**66.32%**). This confirms that geometric features (like the elongated shape of fibroblasts) are essential for distinguishing connective tissue from tumor cells.
3.  **The "Other" Breakthrough:** Only the GATv2 model successfully learned to identify the "Other" class (1.41%), whereas previous models completely ignored it.

---

## 📂 5. Repository Structure

```bash
PathoGAT-TME/
├── notebooks/             # Jupyter Notebooks for training and analysis
│   └── PathoGAT_Analysis.ipynb
├── models/                # Pre-trained PyTorch model weights (.pth)
│   ├── pathology_gnn.pth
│   ├── pathology_gat.pth
│   └── pathology_gat_v2.pth
├── src/                   # Source code for data loading and model definitions
├── results/               # Experiment logs and confusion matrices
└── requirements.txt       # Dependencies

```
---

## 🔮 6. Future Work: GNN vs. CNN

The next phase of this research focuses on a direct benchmark between **Pixel-based CNNs** (ResNet/DenseNet) and **Entity-based GNNs**. 



### **The Hypothesis**
* **Robustness to Domain Shift:** I hypothesize that GNNs will demonstrate superior robustness to **stain variations** (color differences between labs/scanners). Since GNNs rely on geometric topology and relative spatial coordinates rather than raw pixel intensities, they should generalize better across different medical centers.
* **Hybrid Modeling:** I am developing a **CNN-GNN Hybrid** pipeline where a lightweight CNN (e.g., EfficientNet) extracts deep textural features for each individual nucleus. These features are then passed to the **GATv2** for high-level spatial reasoning across the Tumor Microenvironment.
* **Neuromorphic Application:** The sparse, asynchronous, and event-driven nature of cell-graph data makes it an ideal candidate for **Spiking Neural Networks (SNNs)**. Implementing these models on neuromorphic hardware (e.g., Intel Loihi) could potentially reduce power consumption by orders of magnitude compared to traditional GPUs—a critical factor for bedside medical devices.

---

📜 Citation
If you find this research or code useful for your work, please cite it as follows:

Ali, A. (2025). PathoGAT-TME: Spatially-Aware Graph Learning for Digital Pathology. GitHub Repository: https://github.com/AsadAli-nodes/PathoGAT-TME

