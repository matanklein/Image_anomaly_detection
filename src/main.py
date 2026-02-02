import os
import argparse
from preprocess import process_pcap_to_summed_images
from train import train_model
from test import test_model 
import config

def ensure_dirs():
    """Ensure all required directories exist for storing tensors and model outputs."""
    # Create base tensor directory
    os.makedirs(config.TENSORS_DIR, exist_ok=True)
    os.makedirs(config.TRAIN_BENIGN_DIR, exist_ok=True)
    os.makedirs(config.TRAIN_OE_DIR, exist_ok=True)
    # Create test directories
    os.makedirs(config.TEST_BENIGN_DIR, exist_ok=True)
    os.makedirs(config.TEST_MALICIOUS_DIR, exist_ok=True)

def run_single_attack(attack, args):
    """
    Helper function to update config and run the pipeline for one attack.
    Respects args.preprocess and args.test flags.
    """
    attack_name = attack['name']
    print(f"\n>>> Setting up environment for: {attack_name}")
    
    # 1. DYNAMICALLY UPDATE CONFIG
    config.PCAP_PATH = attack['pcap']
    config.ATTACKER_IP = attack['attacker_ip']
    config.VICTIM_IP = attack['victim_ip']
    
    # Update Output Directory to avoid overwriting
    base_test_dir = f"{config.TENSORS_DIR}/test/{attack_name}"
    config.TEST_BENIGN_DIR = f"{base_test_dir}/benign"
    config.TEST_MALICIOUS_DIR = f"{base_test_dir}/malicious"
    
    # 2. CREATE DIRS
    ensure_dirs()
    
    # 3. OPTIONAL: RUN PREPROCESS (PCAP -> Images)
    if args.preprocess:
        print(f"   [1/2] Processing PCAP...")
        try:
            process_pcap_to_summed_images(mode='test') 
        except Exception as e:
            print(f"Error in processing: {e}")
            return # Stop if preprocessing fails
    else:
        print(f"   [1/2] Skipping Preprocessing (images must already exist)...")

    # 4. OPTIONAL: RUN INFERENCE
    if args.test:
        print(f"   [2/2] Running Evaluation...")
        try:
            test_model()
        except Exception as e:
            print(f"Error in testing: {e}")
    else:
        print(f"   [2/2] Skipping Test...")

def get_batch_attacks(dataset_name):
    """
    Returns the list of attacks based on the selected dataset argument.
    Assumes the external files define a variable named 'ATTACKS'.
    """
    try:
        if dataset_name == 'cic-ids-2018':
            from cic_2018_details import ATTACKS
            return ATTACKS
            
        # elif dataset_name == 'ton-iot':
        #     from ton_iot_details import ATTACKS
        #     return ATTACKS
            
        else:
            print(f"Error: Unknown dataset name '{dataset_name}'. Cannot load attack list.")
            return []
            
    except ImportError as e:
        print(f"Error: Could not import attack details for {dataset_name}.")
        print(f"   Make sure the file exists and has an 'ATTACKS' list variable.")
        print(f"   Python Error: {e}")
        return []
     
def main(args):
    # --- MODE 1: BATCH BENCHMARK ---
    if args.benchmark:
        print("STARTING BATCH BENCHMARK MODE")
        
        # Safety Check
        if args.train:
            print("ERROR: --benchmark cannot be combined with --train.")
            print("   Benchmark is for testing existing models only.")
            return
        
        if not (args.preprocess or args.test):
            print("WARNING: You selected --benchmark but didn't specify --preprocess or --test.")
            print("   Nothing will happen. Please add flags.")
            return
        
        # 1. Load the specific attacks for this dataset
        batch_attacks = get_batch_attacks(args.dataset)
        
        if not batch_attacks:
            print("WARNING: No attacks found. Exiting.")
            return
        
        print("="*40)
        for i, attack in enumerate(batch_attacks):
            print(f"Experiment {i+1}/{len(batch_attacks)}")
            run_single_attack(attack, args)
            
        print("\nBatch Benchmark Completed.")
        return

    # --- MODE 2: STANDARD SINGLE RUN ---
    print("Step 1: Ensuring required folders exist...")
    ensure_dirs()

    if args.preprocess and args.train:
        print("Step 2: Processing flows and converting to image tensors...")
        process_pcap_to_summed_images('train')
        
        print("Step 3: Training model on benign traffic...")
        train_model()


    elif args.preprocess and args.test:
        print("Step 2: Processing flows and converting to image tensors...")
        process_pcap_to_summed_images('test')

        print("Step 3: Testing model on mixed traffic...")
        test_model()
    
    elif args.test:
        print("Step 2: Testing model on mixed traffic...")
        test_model() 
    
    elif args.train:
        print("Step 2: Training model on benign traffic...")
        train_model()
        
    if not any([args.preprocess, args.train, args.test]):
        print("No action specified. Use --preprocess, --train, or --test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Graph CNN Pipeline")
    parser.add_argument('--preprocess', action='store_true', help="Run flow processing and graph-to-image conversion")
    parser.add_argument('--train', action='store_true', help="Train CNN model on benign traffic")
    parser.add_argument('--test', action='store_true', help="Test model on mixed traffic")

    parser.add_argument('--benchmark', action='store_true', help="Run ALL attacks defined in BATCH_ATTACKS list")
    parser.add_argument('--dataset', type=str, default='cic-2018', help="Name of the dataset (e.g., ton-iot, cic-ids-2018, cic-mal-anal).")
    
    args = parser.parse_args()
    main(args)