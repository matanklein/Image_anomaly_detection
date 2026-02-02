import os
import numpy as np
from scapy.all import IP, TCP, UDP
from scapy.utils import PcapReader
from collections import defaultdict
from tqdm import tqdm
import config

# Custom FlowPic generation function
def get_flowpic(timetofirst, pkts_size, flowpic_dim=1500, max_block_duration=60):
    """
    Generates a FlowPic (2D Histogram) from packet time-series data.
    
    This function maps packet arrivals into a 2D grid where:
    - X-axis represents Packet Size (normalized 0 to max_pkts_size).
    - Y-axis represents Arrival Time (normalized 0 to max_block_duration).
    - Pixel intensity represents the count of packets in that bin.

    Args:
        timetofirst (np.ndarray): Array of arrival times (in seconds) relative 
                                  to the first packet of the flow.
        pkts_size (np.ndarray): Array of packet sizes (in bytes).
        flowpic_dim (int, optional): The resolution of the output image (W x H). 
                                     Defaults to 1500.
        max_block_duration (int, optional): The maximum time window (in seconds) 
                                            to consider. Packets arriving after 
                                            this are discarded. Defaults to 60.
    Returns:
        np.ndarray: A 2D numpy array of shape (flowpic_dim, flowpic_dim) with 
                    dtype uint8. Values are clipped to [0, 255].
    """
    # Filter: Ignore packets that arrived after the time window
    # Note: np.where returns a tuple, we take the first element [0]
    valid_indices = np.where(timetofirst < max_block_duration)[0]
    
    # Apply filter
    timetofirst = timetofirst[valid_indices]
    pkts_size = pkts_size[valid_indices]

    # Clip Sizes: Ensure no packet exceeds the theoretical max
    # This prevents index-out-of-bounds errors during histogram generation
    pkts_size = np.clip(pkts_size, a_min=0, a_max=config.MAX_PACKET_SIZE)

    # Normalize coordinates to image dimensions
    # We map [0, max_duration] -> [0, dim]
    # We map [0, max_size]     -> [0, dim]
    timetofirst_norm = (timetofirst / max_block_duration) * flowpic_dim
    pkts_size_norm = (pkts_size / config.MAX_PACKET_SIZE) * flowpic_dim

    # Generate Histogram
    # We use explicit bins defined by range(dim + 1) to align pixels perfectly
    # x=pkts_size (Size Axis), y=timetofirst (Time Axis)
    bins = range(flowpic_dim + 1)
    mtx, _, _ = np.histogram2d(
        x=pkts_size_norm, 
        y=timetofirst_norm, 
        bins=[bins, bins]
    )

    # Post-process based on config
    if config.IMAGE_TYPE == 'binary':
        # Convert counts to binary: >0 becomes 1
        mtx = np.where(mtx > 0, 1, 0)

    elif config.IMAGE_TYPE == 'normal':
        # Cap counts at 255 (uint8 max) to create a valid grayscale image.
        # Any bin with >255 packets will be set to exactly 255.
        mtx = np.clip(mtx, a_min=0, a_max=255).astype("uint8")
    
    return mtx

def get_5tuple(pkt):
    """Extracts src_ip, dst_ip, src_port, dst_port, proto."""
    try:
        ip_layer = pkt[IP]
        proto = ip_layer.proto
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        
        src_port = 0
        dst_port = 0
        
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            
        return (src_ip, dst_ip, src_port, dst_port, proto)
    except IndexError:
        return None

def process_pcap_to_summed_images(mode='train'):
    print(f"--- Processing {config.PCAP_PATH} ---")
    
    # State variables
    current_interval_start = -1
    
    # Dictionary to store flow data: key=5tuple, val=[(ts, size), ...]
    # We store tuples to minimize object overhead compared to storing scapy packets
    active_flows = defaultdict(list)
    
    # Reader
    reader = PcapReader(config.PCAP_PATH)
    
    # Progress bar (approximate since we don't know total packets in stream)
    pbar = tqdm(desc="Processing Packets", unit="pkt")

    label = False  # Default label for the current interval
    
    try:
        for pkt in reader:
            if IP not in pkt:
                continue
            
            pbar.update(1)
            ts = float(pkt.time)
            size = len(pkt)
            
            # Initialize start time
            if current_interval_start == -1:
                current_interval_start = ts

            # --- CHECK TIME INTERVAL ---
            if ts >= current_interval_start + config.FLOWPIC_TIME_INTERVAL:
                
                # PROCESS PREVIOUS INTERVAL
                save_interval(active_flows, current_interval_start, label, mode = mode)
                
                # FLUSH / RESET FOR NEXT INTERVAL
                current_interval_start += config.FLOWPIC_TIME_INTERVAL
                label = False
                
                active_flows.clear()

            # --- ACCUMULATE PACKET ---
            five_tuple = get_5tuple(pkt)
            if five_tuple:
                # Add to flow data
                active_flows[five_tuple].append((ts, size))

                if label == False:
                    if (five_tuple[0] in config.ATTACKER_IP) and (five_tuple[1] in config.VICTIM_IP):
                        label = True
                    elif (five_tuple[0] in config.VICTIM_IP) and (five_tuple[1] in config.ATTACKER_IP):
                        label = True

        # Save the very last interval if it has data
        if active_flows:
            save_interval(active_flows, current_interval_start, label, mode = mode)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.close()
        pbar.close()

def save_interval(active_flows, interval_start_ts, label, mode):
    """
    Generates FlowPics for all flows, sums them, determines label, and saves.
    """
    if not active_flows:
        return

    # Initialize the master summed image
    # Shape: (1, 1500, 1500) assuming grayscale/binary single channel
    summed_image = np.zeros((config.FLOWPIC_DIM, config.FLOWPIC_DIM), dtype=np.float32)

    # --- AGGREGATION LOOP ---
    for flow_key, packets in active_flows.items():
        # Unzip timestamps and sizes
        timestamps, sizes = zip(*packets)
        timestamps = np.array(timestamps)
        sizes = np.array(sizes)
        
        # Calculate Time-To-First (ttft) for this specific flow
        # FlowPic logic relies on time relative to the *start of the flow*
        start_of_flow = timestamps[0]
        timetofirst = timestamps - start_of_flow
        
        # Call tcbench to generate the image
        # Note: get_flowpic might return a tensor or numpy array depending on version
        # We ensure it matches dimensions
        try:
            flowpic = get_flowpic(
                timetofirst=timetofirst,
                pkts_size=sizes,
                flowpic_dim=config.FLOWPIC_DIM,
                max_block_duration=config.FLOWPIC_TIME_INTERVAL
            )
            
            # Ensure format is compatible for addition
            if hasattr(flowpic, 'numpy'):
                flowpic = flowpic.numpy()
            
            # --- SUMMATION ---
            # Add to the grand total picture
            summed_image += flowpic
            
        except Exception as e:
            # Skip flows that fail generation (e.g. single packet flows might trigger edge cases)
            print(f"Warning: Failed to generate FlowPic for flow {flow_key}: {e}")
            continue

    # --- CLIPPING LOGIC ---
    # Validate via config boolean if we should cap values at 255
    # This ensures that even if 500 flows overlap in one bin, the pixel value stays 255
    if config.CLIP_SUMMED_COUNTS:
        summed_image = np.clip(summed_image, 0, 255)

    output_dir = get_output_dir(mode, label)

    # --- SAVING ---
    # Format: {ts}_{label}.npy
    filename = f"{int(interval_start_ts)}_{label}.npy"
    save_path = os.path.join(output_dir, filename)
    
    if config.CLIP_SUMMED_COUNTS:
        # Save as uint8 if clipped
        np.save(save_path, summed_image.astype("uint8"))
    else:
        np.save(save_path, summed_image)

def get_output_dir(mode, label):
    if mode == 'train':
        if label:
            return config.TRAIN_OE_DIR
        else:
            return config.TRAIN_BENIGN_DIR
    
    elif mode == 'test':
        if label:
            return config.TEST_MALICIOUS_DIR
        else:
            return config.TEST_BENIGN_DIR
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
