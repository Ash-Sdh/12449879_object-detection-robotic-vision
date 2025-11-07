## Assignment 2 Summary

**Metric:** mAP@[.5:.95]; **Speed:** latency (ms/img), FPS  
**Dataset used for prototyping:** coco128 (official COCO sample); full COCO planned for later  

**Target:** Baseline mAP ≥ 0.30 (on coco128); CBAM +1–2 mAP with ≤25 % extra latency  

**Achieved:**  
| Model           | mAP@[.5:.95] | Latency (ms/img) | FPS |
|-----------------|--------------|------------------|-----|
| YOLOv5s         | 0.29         | 4.4 ms           | ~227 FPS |
| YOLOv5s + CBAM  | 0.32 (+0.03) | 8.6 ms (+4.2)    | ~116 FPS |

**Notes on trade-off:**  
CBAM improved accuracy by +0.03 mAP but doubled inference time.  
Still >100 FPS, acceptable for mid-speed robotic vision tasks.  

**Time log (based on WBS):**
- Data setup (coco128): 4 h  
- Baseline train + val: 8 h  
- CBAM integration + train: 10 h  
- Speed measurements: 4 h  
- README updates / cleanup: 4 h  
**Total ≈ 30 h**

**Conclusion:**  
Baseline pipeline and CBAM integration both verified on Colab.  
CBAM shows slight accuracy gain at moderate latency cost.

# 12449879_object-detection-robotic-vision

## Object Detection for Robotic Vision using YOLOv5 with Attention Modules (CBAM + Transformers)

---

### 1. Project Idea and Approach

This project aims to enhance visual perception for robotic systems by improving object detection accuracy.  
A pre-trained **YOLOv5** network will be used as the baseline detector on the COCO dataset.  
To achieve better precision in cluttered or complex environments, a **Convolutional Block Attention Module (CBAM)** will be added to the YOLOv5 backbone.  
CBAM allows the model to focus more effectively on relevant spatial and channel features while suppressing less important areas.

An additional exploratory step will include a **light Transformer-based attention layer** to capture global context relationships.  
This hybrid attention approach combines the efficiency of CNN-based YOLO models with the contextual awareness of Transformer architectures.

Model performance will be evaluated using **mean Average Precision (mAP)** and visualized through detection results and attention heatmaps.

---

### 2. Dataset Description

**Dataset:** COCO – Common Objects in Context  
**Source:** [https://cocodataset.org/](https://cocodataset.org/)  
**Content:** 330,000 images with 80 object categories (e.g., person, cup, chair, car, etc.)  
**Annotations:** Bounding boxes and labels for all objects  
**Splits:** 118k training, 5k validation, and 20k test images  
**Evaluation Metric:** mAP @ IoU = 0.5–0.95  


---

### 3. Methodology (Planned Steps)

| Step | Description |
|------|--------------|
| 1 | Set up baseline YOLOv5 model (pre-trained weights on COCO). |
| 2 | Integrate CBAM attention modules into YOLOv5 backbone layers. |
| 3 | Additionally add a small Transformer attention block for feature fusion. |
| 4 | Apply data augmentation (flipping, rotation, brightness adjustments) to improve robustness. |
| 5 | Train and validate the model using mAP and loss curves. |
| 6 | Generate Grad-CAM visualizations for explainability. |
| 7 | Visualize and present detection results |

---

### 4. Work Breakdown Structure (WBS)

| Task | Description | Estimated Hours |
|------|--------------|----------------|
| Dataset collection / pre-processing | Download COCO subset, verify annotations | 4 h |
| Design and build network | Set up YOLOv5 and insert CBAM modules | 8 h |
| Training and validation | Run experiments, log metrics | 10 h |
| Fine-tuning  | Compare baseline vs improved model | 6 h |
| Application / demo | Create inference notebook for robotic vision | 4 h |
| Report writing | Summarize results and prepare documentation | 4 h |
| Presentation prep | Prepare slides and visual examples | 4 h |
| **Total** |  | **40 hours** |

---

### 5. Expected Outcome

- Improved mAP and visual clarity through attention-guided feature extraction  
- Visualization of attention maps to show focus regions
- A final presentation and short written report will summarize the findings.


---

### 6. References Research Papers

- Redmon, J., & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement.* arXiv:1804.02767. [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)  
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module.* In ECCV 2018. [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)  
- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). *End-to-End Object Detection with Transformers (DETR).* arXiv:2005.12872. [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
---

