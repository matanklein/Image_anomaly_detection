# Network Anomaly Detection via Image-Based Deep Learning

🔍 **Detecting network intrusions and anomalies by transforming raw network traffic into spatial image representations and analyzing them using CNN-based Deep Support Vector Data Description (Deep SVDD).**

---

## 📖 Methodology

This project introduces a novel approach to cybersecurity anomaly detection: by modeling network traffic continuously over space and time as "images" (FlowPics), we can leverage powerful computer vision models (Convolutional Neural Networks) to find spatial patterns indicative of cyber attacks. 

The strategy is broken down into two core phases: Image Generation (Data Transformation) and Model Training.

### 🖼️ Data Transformation: PCAP to Image
Instead of relying on domain-specific feature engineering (like packet counts or flow duration metadata), we visualize the direct behavior of network flows. The process goes as follows:

1. **Traffic Capture:** We process raw network traffic either directly from `PCAP` captures or parsed flow CSVs.
2. **5-Tuple Extraction:** For every packet, we dynamically extract the flow identifier consisting of Source IP, Destination IP, Source Port, Destination Port, and Protocol.
3. **FlowPic Generation:** Packets belonging to distinct flows are mapped onto a 2D histogram structure over fixed time intervals (e.g., 60 seconds).
   * **X-Axis:** Represents the packet size in bytes (0 to 1500).
   * **Y-Axis:** Represents the relative arrival time within the time block.
   * **Pixel Intensity:** Signifies the accumulation/count of packets hitting that specific size/time bin.
4. **Aggregation:** Individual FlowPics are aggregated together into a master "summed image" representing the overarching state of network traffic in that time slice. Capping and binarization strategies can be applied to handle varying traffic densities.

By transforming packets into images, sophisticated attack patterns (like high-frequency scanning, slow-rate DDoS, or covert channel communication) appear as distinct visual artifacts.

### 🧠 Model Architecture: CNN & Deep SVDD
Once the traffic is represented as images, we employ a **Convolutional Neural Network (CNN)** paired with a **Deep SVDD (Support Vector Data Description)** objective function to identify malicious behavior.

* **Convolutional Layers:** Extract spatial hierarchies and local topological behaviors directly from the FlowPics. The network compresses the 1500x1500 image input down to a lower-dimensional latent representation (e.g., 64 dimensions).
* **Deep SVDD Paradigm:** Traditional supervised learning requires perfectly labeled attack data. Instead, Deep SVDD performs semi-supervised anomaly detection. 
* The model maps normal ("benign") traffic instances as close as possible to a single compact hypersphere center `c` in the latent space.
* We compute a threshold radius around `c`. 
* If Outlier Exposure (OE) samples or recognized malicious attacks are introduced during training, they are explicitly penalized and pushed outside a predefined distance margin.
* During inference, if the distance of a new image from `c` exceeds the calibrated threshold margin, it is flagged as anomalous.

---

## ⚙️ Pipeline Flow

Here is the step-by-step technical pipeline:

1. **Data Ingestion** (`Raw Traffic`) -> Read sequential packets via Scapy / CSV Reader.
2. **Interval Batching** (`Buffering`) -> Group packets into discrete time windows (default 60s).
3. **Spatial Mapping** (`Image Generation`) -> Map Times and Sizes onto a 2D grid matrix per flow. 
4. **Summation** (`Aggregated Snapshot`) -> Combine overlapping flows into a singular tensor representing network state.
5. **CNN Embedding** (`Feature Extraction`) -> Pass tensor into the CNN layer stack to generate a 64d latent vector.
6. **Distance Calculation** (`Scoring`) -> Compute Euclidean distance of the embedding relative to the learned normal center `c`.
7. **Classification** (`Anomaly Warning`) -> Flag as an Intrusion if distance `> Threshold`, otherwise predict Benign.

---

## 🛠️ Setup & Usage

### 📦 Prerequisites
Install the required packages. Ensure you have PyTorch set up corresponding to your CUDA environment for hardware acceleration.

```bash
pip install torch torchvision scapy numpy tqdm
```

### 🚀 CLI Execution Reference

The entry point for all operations is `src/main.py`. The pipeline execution is modularized and relies on parameters configured inside `src/config.py`.

**1. Full Pipeline (Preprocess, Train, Test)**
Run processing to build tensors, train the Deep SVDD model on benign traffic (optionally penalized by Outlier Exposure data), and test on an unseen mixed dataset.
```bash
python src/main.py --preprocess --train --test
```

**2. Generate Images Only (Preprocess Data)**
Convert the PCAP or CSV files specified in `config.py` into NumPy Tensor representations.
```bash
python src/main.py --preprocess
```

**3. Train the Model Only**
Train a new CNN Deep SVDD model assuming the underlying tensors have already been generated and reside in the configured output directory.
```bash
python src/main.py --train
```

**4. Inference / Test Mode**
Evaluate an existing trained model on test tensors to output predictions and classification metrics.
```bash
python src/main.py --test
```

**5. Benchmark Mode / Batch Testing**
Run the entire anomaly detection evaluation consecutively across multiple different attack profiles specified for a given dataset (e.g., CIC-IDS-2018).
```bash
python src/main.py --benchmark --dataset cic-ids-2018
```

---

## 📊 Performance Metrics

| Metric | Score | Description |
|---|---|---|
| **Accuracy**  | `XX.X%` | Overall correctness of benign vs malicious classifications. |
| **Precision** | `XX.X%` | Ratio of true detected attacks to all flagged anomalies. |
| **Recall**    | `XX.X%` | Ratio of detected attacks to the actual total number of attacks. |
| **F1-Score**  | `XX.X%` | Harmonic mean of Precision and Recall. Robustness indicator. |
