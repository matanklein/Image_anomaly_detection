import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scapy.all import PcapReader, IP
from tqdm import tqdm
from Image_anomaly_detection.src.config import Config

# Helper function to mimic tcbench's get_flowpic logic but optimized for direct summing
def generate_histogram(timestamps, sizes, dim=1500, max_duration=60):
    """
    Generates a 2D histogram (FlowPic) from raw packet data.
    """
    if not timestamps:
        return np.zeros((dim, dim), dtype=np.float32)
    
    # Normalize times (0 to max_duration) and sizes (0 to 1500)
    times = np.array(timestamps)
    times = times - times[0] # Relative to first packet in window
    
    # Binning
    # Size bins: 0..1500 (MTU)
    # Time bins: 0..max_duration
    H, _, _ = np.histogram2d(
        times, sizes, 
        bins=[dim, dim], 
        range=[[0, max_duration], [0, 1500]]
    )
    
    return H

class PcapFlowPicDataset(Dataset):
    def __init__(self, pcap_path, window_size=60, img_dim=1500, img_type='summed', malicious_ips=None):
        self.img_dim = img_dim
        self.img_type = img_type
        self.window_size = window_size
        self.malicious_ips = set(malicious_ips) if malicious_ips else set()
        
        self.samples = []
        self._preprocess_pcap(pcap_path)

    def _preprocess_pcap(self, pcap_path):
        print(f"Preprocessing {pcap_path}...")
        
        current_window_start = -1
        window_packets_ts = []
        window_packets_len = []
        is_window_malicious = False
        
        try:
            with PcapReader(pcap_path) as pcap:
                for pkt in tqdm(pcap, desc="Reading Packets"):
                    if IP not in pkt:
                        continue
                        
                    ts = float(pkt.time)
                    length = len(pkt)
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    
                    if current_window_start == -1:
                        current_window_start = ts
                    
                    # Check if window closed
                    if ts >= current_window_start + self.window_size:
                        # Save window data
                        self.samples.append({
                            'ts': window_packets_ts,
                            'len': window_packets_len,
                            'label': 1 if is_window_malicious else 0,
                            'start_time': current_window_start
                        })
                        
                        # Reset for next window
                        current_window_start = current_window_start + self.window_size
                        # Fast forward empty windows if any
                        while ts >= current_window_start + self.window_size:
                             current_window_start += self.window_size
                             
                        window_packets_ts = []
                        window_packets_len = []
                        is_window_malicious = False
                    
                    # Add packet to current window
                    window_packets_ts.append(ts)
                    window_packets_len.append(length)
                    
                    # Check label (if ANY packet in window is malicious, label window as anomaly)
                    if src_ip in self.malicious_ips or dst_ip in self.malicious_ips:
                        is_window_malicious = True
                        
        except Exception as e:
            print(f"Warning reading PCAP: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        
        # Create FlowPic
        # The "Summation" is implicit here because we are histogramming ALL packets 
        # in the time interval, regardless of flow ID.
        image = generate_histogram(
            data['ts'], 
            data['len'], 
            dim=self.img_dim, 
            max_duration=self.window_size
        )
        
        # Transform
        if self.img_type == 'binary':
            image = np.where(image > 0, 1.0, 0.0).astype(np.float32)
        else:
            # Grayscale / Summed
            image = np.log1p(image) # Log scaling usually helps with count data
            image = image / image.max() if image.max() > 0 else image # Normalize 0-1
            image = image.astype(np.float32)
            
        # Add channel dim: [C, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        label_tensor = torch.tensor(data['label'], dtype=torch.long)
        
        return image_tensor, label_tensor