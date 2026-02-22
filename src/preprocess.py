import os
import csv
import numpy as np
from datetime import datetime
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

    elif config.IMAGE_TYPE == 'limited_count':
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

def process_pcap_to_summed_images(pcap_path, mode='train'):
    print(f"--- Processing {pcap_path} ---")
    
    # State variables
    current_interval_start = -1
    
    # Dictionary to store flow data: key=5tuple, val=[(ts, size), ...]
    # We store tuples to minimize object overhead compared to storing scapy packets
    active_flows = defaultdict(list)
    
    # Reader
    reader = PcapReader(pcap_path)
    
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

def _first_nonempty(row, keys, default=""):
    for key in keys:
        if key in row:
            value = row[key]
            if value is not None and str(value).strip() != "":
                return value
    return default

def _safe_int(value, default=0):
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError, AttributeError):
        return default

def _safe_float(value, default=None):
    try:
        return float(str(value).strip())
    except (ValueError, TypeError, AttributeError):
        return default

def _normalize_row_keys(row):
    normalized = {}
    for key, value in row.items():
        if key is None:
            continue
        clean_key = str(key).replace("\ufeff", "").strip().strip('"').strip("'").lower()
        if isinstance(value, str):
            value = value.replace("\ufeff", "").strip().strip('"').strip("'")
        normalized[clean_key] = value
    return normalized

def _parse_timestamp(row):
    ts_raw = _first_nonempty(row, [
        "frame.time_epoch",
        "frame.time",
        "timestamp",
        "time",
        "ts",
    ])

    if ts_raw is None:
        return None

    text = str(ts_raw).strip()
    if text == "":
        return None

    # Fast path: epoch seconds string/number
    direct = _safe_float(text)
    if direct is not None:
        return direct

    # Handle common Wireshark timestamp strings, e.g.:
    # "Feb 15, 2018 11:59:01.123456000 IST"
    # "2018-02-15 11:59:01.123456"
    try:
        cleaned = " ".join(text.split())
        for fmt in (
            "%b %d, %Y %H:%M:%S.%f",
            "%b %d, %Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(cleaned, fmt).timestamp()
            except ValueError:
                continue
    except Exception:
        pass

    return None

def get_5tuple_from_csv_row(row):
    """Extracts src_ip, dst_ip, src_port, dst_port, proto from a CSV row."""
    src_ip = _first_nonempty(row, ["ip.src", "src_ip", "source_ip", "source", "src"]).strip()
    dst_ip = _first_nonempty(row, ["ip.dst", "dst_ip", "destination_ip", "destination", "dst"]).strip()

    proto = _safe_int(_first_nonempty(row, ["ip.proto", "protocol", "proto"], default="0"), default=0)

    tcp_sport = _safe_int(_first_nonempty(row, ["tcp.srcport", "tcp.sport", "srcport", "source.port", "src_port"], default="0"), default=0)
    tcp_dport = _safe_int(_first_nonempty(row, ["tcp.dstport", "tcp.dport", "dstport", "destination.port", "dst_port"], default="0"), default=0)
    udp_sport = _safe_int(_first_nonempty(row, ["udp.srcport", "udp.srcpor", "udp.sport", "srcport", "src_port"], default="0"), default=0)
    udp_dport = _safe_int(_first_nonempty(row, ["udp.dstport", "udp.dstpor", "udp.dport", "dstport", "dst_port"], default="0"), default=0)

    if proto == 6:
        src_port, dst_port = tcp_sport, tcp_dport
    elif proto == 17:
        src_port, dst_port = udp_sport, udp_dport
    else:
        src_port, dst_port = (tcp_sport or udp_sport), (tcp_dport or udp_dport)

    if src_ip == "" or dst_ip == "":
        return None

    return (src_ip, dst_ip, src_port, dst_port, proto)

def process_csv_to_summed_images(csv_path, mode='train'):
    """
    Processes packet CSV rows into interval-level summed FlowPics.

    Expected columns include:
    - ip.src, ip.dst, ip.proto, frame.len, frame.time
    - tcp.srcport, tcp.dstport, udp.srcport, udp.dstport (when relevant)
    """

    print(f"--- Processing {csv_path} ---")

    current_interval_start = -1
    active_flows = defaultdict(list)
    label = False
    rows_total = 0
    rows_skipped_ts = 0
    rows_skipped_5tuple = 0
    rows_used = 0
    intervals_saved = 0

    try:
        with open(csv_path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            print(f"CSV columns detected ({len(reader.fieldnames or [])}): {reader.fieldnames}")
            pbar = tqdm(desc="Processing CSV Rows", unit="row")

            for row in reader:
                pbar.update(1)
                rows_total += 1
                row = _normalize_row_keys(row)

                ts = _parse_timestamp(row)
                size = _safe_int(_first_nonempty(row, ["frame.len", "length", "pkt_len", "packet_length"], default="0"), default=0)

                if ts is None:
                    rows_skipped_ts += 1
                    continue

                if current_interval_start == -1:
                    current_interval_start = ts

                if ts >= current_interval_start + config.FLOWPIC_TIME_INTERVAL:
                    if save_interval(active_flows, current_interval_start, label, mode=mode):
                        intervals_saved += 1

                    current_interval_start += config.FLOWPIC_TIME_INTERVAL
                    label = False
                    active_flows.clear()

                five_tuple = get_5tuple_from_csv_row(row)
                if five_tuple:
                    rows_used += 1
                    active_flows[five_tuple].append((ts, size))

                    if label is False:
                        if (five_tuple[0] in config.ATTACKER_IP) and (five_tuple[1] in config.VICTIM_IP):
                            label = True
                        elif (five_tuple[0] in config.VICTIM_IP) and (five_tuple[1] in config.ATTACKER_IP):
                            label = True
                else:
                    rows_skipped_5tuple += 1

            if active_flows:
                if save_interval(active_flows, current_interval_start, label, mode=mode):
                    intervals_saved += 1

            pbar.close()
            print("--- CSV processing summary ---")
            print(f"Rows read:            {rows_total}")
            print(f"Rows skipped (time):  {rows_skipped_ts}")
            print(f"Rows skipped (tuple): {rows_skipped_5tuple}")
            print(f"Rows used:            {rows_used}")
            print(f"Intervals saved:      {intervals_saved}")
            print(f"Output base dir:      {config.TENSORS_DIR}")

    except KeyboardInterrupt:
        print("\nStopping...")

def save_interval(active_flows, interval_start_ts, label, mode):
    """
    Generates FlowPics for all flows, sums them, determines label, and saves.
    """
    if not active_flows:
        return False

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
    os.makedirs(output_dir, exist_ok=True)

    # --- SAVING ---
    # Format: {ts}_{label}.npy
    filename = f"{int(interval_start_ts)}_{label}.npy"
    save_path = os.path.join(output_dir, filename)
    
    if config.CLIP_SUMMED_COUNTS:
        # Save as uint8 if clipped
        np.save(save_path, summed_image.astype("uint8"))
    else:
        np.save(save_path, summed_image)

    print(f"Saved interval image: {save_path} | flows={len(active_flows)} | label={'malicious' if label else 'benign'}")
    return True

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

def process_source_to_summed_images(mode='train'):
    """
    Main entry point to process the configured SOURCE_PATH into summed images.
    """
    if config.SOURCE_PATH.endswith('.pcap'):
        process_pcap_to_summed_images(config.SOURCE_PATH, mode=mode)
    elif config.SOURCE_PATH.endswith('.csv'):
        process_csv_to_summed_images(config.SOURCE_PATH, mode=mode)
    else:
        raise ValueError(f"Unsupported file type for SOURCE_PATH: {config.SOURCE_PATH}")
