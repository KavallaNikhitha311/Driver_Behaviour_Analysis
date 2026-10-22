# 🚗 Driver Behaviour Detection System for Intelligent Vehicle Safety

An end-to-end, real-time **Driver Behaviour Detection System** designed for Advanced Driver Assistance Systems (ADAS). Powered by deep transfer learning on **EfficientNetV2-S**, the system detects distracted driver states, explains predictions via **Grad-CAM**, filters temporal noise, and triggers in-cabin **Safety HUD warnings** with automated telemetry logging.

---

## 🌟 Key Features

- **High-Performance Backbone:** Fine-tuned `EfficientNetV2-S` trained on 10 driver behavior classes, capturing fine-grained hand positions and device interactions.
- **Explainable AI (XAI):** Integrated **Grad-CAM** visualizer generating class-activation heatmaps to verify focus on steering wheels, phones, and vehicle controls.
- **Temporal Smoothing Engine:** Rolling window probability buffer (`deque`) and consecutive-frame confirmation logic (8 frames) to eliminate single-frame flickering and false alarms.
- **Three-Tier Risk Assessment:**
  - 🟢 **LOW Risk:** Safe driving (`c0`)
  - 🟠 **MEDIUM Risk:** Operating radio, drinking, talking to passenger (`c5`, `c6`, `c9`)
  - 🔴 **HIGH Risk:** Texting, talking on phone, reaching behind, makeup (`c1`–`c4`, `c7`, `c8`)
- **Automated Vehicle Safety HUD:** Real-time OpenCV heads-up display overlay showing risk level, behavior confidence bars, live FPS, and flashing critical warning banners.
- **Telemetry Event Logging:** Automated export to `driver_event_log.csv` tracking timestamps, frame indices, confidence scores, and safety alert states.

---

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
