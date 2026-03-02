# Software Requirements Specification (SRS) & Testing Plan

**System:** RS-Personalized AI-Driven Adaptive Stimulation Software Platform  
**Subsystem:** Aim 2 (Algorithmic & Control Software)  
**Hardware Target:** Off-device Compute (PC/Smartphone) interfacing with Spark Biomedical Sparrow Link.

---

## 1. Specific Functional Requirements (FRs)

These dictate exactly what the software must **DO**. Every requirement here must be testable.

### FR-1.0: Data Ingestion & Streaming

| ID | Requirement |
|----|-------------|
| **FR-1.1** | The system shall ingest real-time physiological data streams (ECG/PPG) from the Galea LSL (Lab Streaming Layer) at a minimum sampling rate of **250 Hz** to ensure accurate HRV calculation. |
| **FR-1.2** | The system shall support bulk ingestion of historical EDF/CSV files from PhysioNet (MIT-BIH, Normal Sinus) and MIMIC-III for offline training. |

### FR-2.0: Signal Preprocessing Pipeline

| ID | Requirement |
|----|-------------|
| **FR-2.1** | The system shall apply a Continuous Wavelet Transform (CWT) to filter baseline wander and high-frequency noise from raw inputs. |
| **FR-2.2** | The system shall extract time-domain (e.g., RMSSD, SDNN) and frequency-domain (LF/HF ratio) HRV indices over rolling **30-second to 5-minute** windows. |

### FR-3.0: Hybrid AI Inference Engine

| ID | Requirement |
|----|-------------|
| **FR-3.1 (CNN)** | The CNN module shall ingest 10-second waveform arrays and output a feature vector representing morphological anomalies. |
| **FR-3.2 (RNN/GRU)** | The RNN module shall ingest the last 5 minutes of HRV features to output a temporal state prediction (Risk of AFib onset). |
| **FR-3.3 (Transformer)** | The Transformer module shall ingest the combined outputs of the CNN and RNN to output optimal stimulation timing (Start/Stop) and intensity. |

### FR-4.0: Hardware Actuation (The "Closed Loop")

| ID | Requirement |
|----|-------------|
| **FR-4.1** | Upon a "Stimulate" decision from the AI ensemble, the system shall generate a digital command packet formatted for the Sparrow Link API. |
| **FR-4.2 (Safety Interlock)** | The system shall **NEVER** command a stimulation duration exceeding FDA safety limits for the Sparrow Link, regardless of AI output. |

---

## 2. Specific Non-Functional Requirements (NFRs)

These dictate **HOW WELL** the system must perform.

### NFR-1.0: Latency & Real-Time Performance

| ID | Requirement |
|----|-------------|
| **NFR-1.1** | The total computational pipeline (Data Ingestion → Preprocessing → AI Inference → API Command Generation) must execute in **< 200 milliseconds** measured at the 99th percentile. |

### NFR-2.0: AI Model Performance Thresholds

| ID | Requirement |
|----|-------------|
| **NFR-2.1** | The classification models shall achieve an Area Under the Receiver Operating Characteristic Curve (AUROC) of **≥ 0.75** in Phase I and **≥ 0.90** in Phase II. *(Note: AUROC is the industry standard for medical AI, not just raw "accuracy", because it accounts for false positives/negatives.)* |

### NFR-3.0: Generalizability (Cross-Domain)

| ID | Requirement |
|----|-------------|
| **NFR-3.1** | When validating a model trained on PhysioNet against the independent MIMIC-III dataset, the F1-Score shall degrade by **no more than 5%**. |

---

## 3. Testing & Validation Requirements (TRs)

As an Architect, proof that the system works is required before it ever touches a human. This is how we prove it.

### Phase A: Data & Model Unit Testing (Months 1–3)

| ID | Test | Acceptance Criteria |
|----|------|---------------------|
| **TR-1.1 (Dataset Partitioning)** | All models must be trained using a strict holdout methodology. | **70%** Training, **15%** Validation, **15%** Blind Testing. The AI must never be tested on data it has "seen" before. |
| **TR-1.2 (K-Fold Cross Validation)** | To ensure the ≥90% accuracy isn't a fluke. | The model must pass a **5-fold cross-validation** test across the MIT-BIH dataset. |
| **TR-1.3 (Noise Injection Test)** | Simulate real-world artifact conditions. | Artificially inject "motion artifact" noise (simulating a patient running or chewing) into the test dataset. The AI must maintain at least **80% accuracy** during simulated noise events to pass. |

### Phase B: System Integration Testing (Months 4–5)

| ID | Test | Acceptance Criteria |
|----|------|---------------------|
| **TR-2.1 (Latency Profiling)** | Verify end-to-end pipeline timing. | Run a script that pushes **10,000 simulated heartbeats** through the Python pipeline. A profiler will measure the exact execution time to verify **NFR-1.1 (< 200 ms)**. |
| **TR-2.2 (Hardware-in-the-Loop Simulation)** | Verify safety interlocks under sustained operation. | Build a "Mock" Sparrow Link receiver in software. The AI will run for **24 simulated hours**, and we will log every command it sends to ensure it **never violates the safety interlocks (FR-4.2)**. |
| **TR-2.3 (Dimensionality Reduction Test)** | Verify robustness to reduced sensor data. | Train the model on the 4K Galea data, then run a test **stripping away 80% of the data channels** (simulating the cheap commercial earbud). The model must still hit the **75% baseline accuracy**. |

---

## Quick Reference: Requirement IDs

| Category | IDs |
|----------|-----|
| **Functional** | FR-1.1, FR-1.2, FR-2.1, FR-2.2, FR-3.1, FR-3.2, FR-3.3, FR-4.1, FR-4.2 |
| **Non-Functional** | NFR-1.1, NFR-2.1, NFR-3.1 |
| **Testing** | TR-1.1, TR-1.2, TR-1.3, TR-2.1, TR-2.2, TR-2.3 |
