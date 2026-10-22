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

---

## 🏗️ Architecture Pipeline

[Input Video / Camera] │ ▼ [Frame Preprocessing (224x224 RGB)] │ ▼ [EfficientNetV2-S Feature Extractor] │ ▼ [Dense Softmax Classification (10 Classes)] │ ▼ [Temporal Smoothing & Moving-Average Buffer] │ ▼ [Dynamic Risk Assessment Engine (LOW / MED / HIGH)] │ ├──► [Real-Time HUD Video: annotated_safety_output.mp4] ├──► [Visual Explainability: Grad-CAM Attention Heatmaps] └──► [Telemetry Blackbox: driver_event_log.csv]


---

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
