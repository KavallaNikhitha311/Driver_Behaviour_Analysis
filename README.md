# 🚗 Driver Behavior & Distraction Analysis for Intelligent Vehicle Safety

An end-to-end, real-time **Driver Monitoring System (DMS)** designed for Advanced Driver Assistance Systems (ADAS). Powered by deep transfer learning on **EfficientNetV2-S**, the system detects driver distraction states, explains predictions via **Grad-CAM**, filters noise with **temporal smoothing**, and triggers real-time in-cabin **Safety HUD alerts** with automated telemetry logging.

---

## 🌟 Key Features

- **High-Performance Backbone:** Fine-tuned `EfficientNetV2-S` trained on 10 driver behavior classes, capturing fine-grained hand positions and device interactions.
- **Explainable AI (XAI):** Integrated **Grad-CAM** visualizer generating class-activation heatmaps to verify focus on steering wheels, phones, and mirrors.
- **Temporal Smoothing Engine:** Rolling window probability buffer (`deque`) and consecutive-frame confirmation logic (8 frames) to completely eliminate single-frame flickering and false alarms.
- **Three-Tier Risk Assessment:**
  - 🟢 **LOW Risk:** Safe driving (`c0`)
  - 🟠 **MEDIUM Risk:** Operating radio, drinking, talking to passenger (`c5`, `c6`, `c9`)
  - 🔴 **HIGH Risk:** Texting, talking on phone, reaching behind, makeup (`c1`–`c4`, `c7`, `c8`)
- **Automated Vehicle Safety HUD:** Real-time OpenCV heads-up display overlay showing risk level, behavior confidence bars, live FPS, and flashing critical warning banners.
- **Telemetry Event Logging:** Automated export to `driver_event_log.csv` tracking timestamps, frame indices, confidence scores, and safety alert states.

```

## 🏗️ Architecture Pipeline

### 1. Visual End-to-End Pipeline

```mermaid
flowchart TD
    subgraph S1["Stage 1: Video Ingestion & Preprocessing"]
        A["In-Cabin Camera / Video Stream"] --> B["OpenCV Frame Extraction (25 FPS)"]
        B --> C["Color Conversion & Resize (224x224 RGB)"]
        C --> D["Tensor Conversion & Batch Prefetch"]
    end

    subgraph S2["Stage 2: Deep Learning Classification"]
        D --> E["EfficientNetV2-S Backbone"]
        E --> F["Global Average Pooling 2D"]
        F --> G["Dropout Regularization (0.3)"]
        G --> H["Dense Softmax Classifier (10 Classes)"]
    end

    subgraph S3["Stage 3: Temporal Smoothing & Risk Engine"]
        H --> I["Rolling Probability Buffer (deque maxlen=12)"]
        I --> J["Moving Average Probability Computation"]
        J --> K["Consecutive High-Risk Filter (8 Frames)"]
        K --> L{"Dynamic Risk Evaluator"}
        L -->|"Class c0"| R1["LOW RISK (Green: Normal Attentive)"]
        L -->|"Classes c5, c6, c9"| R2["MEDIUM RISK (Amber: Caution)"]
        L -->|"Classes c1-c4, c7, c8"| R3["HIGH RISK (Red: Distracted)"]
    end

    subgraph S4["Stage 4: Multi-Modal Output & Telemetry"]
        E -.->|"Feature Maps"| X1["Grad-CAM Explainability Engine"]
        X1 --> X2["Attention Heatmap Visualization"]

        R1 --> HUD["OpenCV Real-Time Safety HUD"]
        R2 --> HUD
        R3 --> HUD

        HUD --> V1["Dynamic Status Badge (Risk & FPS)"]
        HUD --> V2["Behavior Confidence Bar"]
        HUD --> V3["Flashing Critical Alert Banner"]

        V1 --> OUT1["Annotated Safety Video (.mp4)"]
        V2 --> OUT1
        V3 --> OUT1

        R1 --> OUT2["Vehicle Telemetry Event Log (.csv)"]
        R2 --> OUT2
        R3 --> OUT2
    end
```

---

### 2. Detailed Pipeline Breakdown

| Pipeline Stage | Module / Component | Technical Implementation & Purpose |
| :--- | :--- | :--- |
| **1. Acquisition & Preprocessing** | `OpenCV` + `tf.data` | Ingests raw video/webcam frames, resizes to $224 \times 224$ RGB, and prefetches batches for sub-millisecond pipeline throughput. |
| **2. Feature Extraction** | `EfficientNetV2-S` Backbone | Pretrained convolutional network with inverted bottleneck blocks, fine-tuned on the upper 120 layers to isolate driver hands, head tilt, and handheld devices. |
| **3. Classification Head** | `GAP` + `Dropout(0.3)` + `Dense(10)` | Aggregates spatial feature maps into a 10-class softmax distribution across safe and distracted driving behaviors. |
| **4. Explainability (XAI)** | `Grad-CAM` | Computes gradients of the target class score with respect to final convolutional layer feature maps, visually proving model attention on phones, wheels, and controls. |
| **5. Temporal Smoothing** | `DriverBehaviorAnalyzer` | Employs a rolling FIFO queue (`deque(maxlen=12)`) to average frame probabilities, preventing erratic label flickers caused by rapid lighting or angle shifts. |
| **6. False-Alarm Suppression** | 8-Frame Persistence Filter | Driver must continuously exhibit high-risk behavior for $\ge 8$ consecutive frames before triggering an active alert, safely filtering out brief mirror glances. |
| **7. Safety HUD Generation** | `draw_safety_hud` | Renders a high-tech dashboard HUD with color-coded risk badges, behavior confidence bars, live FPS telemetry, and flashing bottom alert banners. |
| **8. Telemetry Event Logging** | `driver_event_log.csv` | Automatically records ISO timestamps, frame indices, predicted behaviors, confidence metrics, and active alert statuses for safety auditing. |

## 📊 Dataset & Target Behaviors (State Farm Dataset)

| Class | Description | Risk Level |
| :--- | :--- | :---: |
| `c0` | Safe driving | 🟢 LOW |
| `c1` | Texting - right hand | 🔴 HIGH |
| `c2` | Talking on phone - right hand | 🔴 HIGH |
| `c3` | Texting - left hand | 🔴 HIGH |
| `c4` | Talking on phone - left hand | 🔴 HIGH |
| `c5` | Operating radio | 🟠 MEDIUM |
| `c6` | Drinking | 🟠 MEDIUM |
| `c7` | Reaching behind | 🔴 HIGH |
| `c8` | Hair and makeup | 🔴 HIGH |
| `c9` | Talking to passenger | 🟠 MEDIUM |

---

## 🛠️ Tech Stack

- **Deep Learning Framework:** TensorFlow 2.x / Keras 3
- **Backbone Architecture:** EfficientNetV2-S (Pretrained on ImageNet)
- **Computer Vision:** OpenCV (cv2)
- **Explainable AI:** Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Data Engineering:** `tf.data` API with mixed precision (`mixed_float16`)
- **Video Encoding:** FFmpeg (`libx264` H.264 web standard)
- **Metrics & Logging:** Pandas, Scikit-Learn, Seaborn, Matplotlib

---

## 📁 Output Deliverables

- `best_driver_behavior_model.keras` — Checkpointed deep learning model.
- `driver_safety_h264.mp4` — Full video with real-time Safety HUD and alert banners.
- `driver_event_log.csv` — Vehicle telemetry log for post-drive safety audits.
