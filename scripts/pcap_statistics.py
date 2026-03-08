from scapy.all import PcapReader, IP, TCP, UDP
import numpy as np
from collections import defaultdict

def analyze_pcap(pcap_file):
    print(f"Analyzing {pcap_file}...\n")
    
    # Dictionary to store flow data. The key will be the 5-Tuple.
    flows = defaultdict(lambda: {'packet_lengths': [], 'timestamps': []})
    
    # Using PcapReader to read the file packet by packet (saves memory compared to rdpcap).
    with PcapReader(pcap_file) as pcap:
        for packet in pcap:
            # We focus only on packets that have an IP layer and TCP/UDP.
            if IP in packet and (TCP in packet or UDP in packet):
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                proto = packet[IP].proto
                
                if TCP in packet:
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                else:
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                
                # Define a unique Flow ID - Unidirectional (Source to Destination)
                flow_id = (src_ip, dst_ip, src_port, dst_port, proto)
                
                # Add packet length and timestamp to the relevant flow
                flows[flow_id]['packet_lengths'].append(len(packet))
                flows[flow_id]['timestamps'].append(float(packet.time))

    # Print headers for the flow statistics
    print(f"{'Source IP':<15} | {'Dest IP':<15} | {'D.Port':<6} | {'Pkts':<5} | {'Duration(s)':<11} | {'Mean Len':<8} | {'Std Dev (Variance)':<18}")
    print("-" * 95)
    
    for flow_id, data in flows.items():
        src_ip, dst_ip, src_port, dst_port, proto = flow_id
        lengths = data['packet_lengths']
        times = data['timestamps']
        
        packet_count = len(lengths)
        
        # Ignore flows that are too short (e.g., just a TCP handshake)
        if packet_count < 3:
            continue 
            
        # Calculate statistics
        flow_duration = max(times) - min(times)
        mean_len = np.mean(lengths)
        
        # This is the critical metric we discussed (variance in XSS vs. brute force)
        std_dev = np.std(lengths) 
        
        # Filter to show only significant flows (e.g., target ports 443, 80, 21, 22)
        if dst_port in [21, 22, 80, 443]:
            print(f"{src_ip:<15} | {dst_ip:<15} | {dst_port:<6} | {packet_count:<5} | {flow_duration:<11.4f} | {mean_len:<8.2f} | {std_dev:<18.2f}")

if __name__ == "__main__":
    # Replace with the path to your PCAP file
    analyze_pcap("../data/cic2018/bruteforce-web-1-22_02_2018.pcap")
    print("Script is ready.")