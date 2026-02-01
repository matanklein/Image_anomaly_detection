# Directory structure
# PCAP_PATH = '../data/cic2018/bruteforce-xss-1-22_02_2018.pcap' # CHANGE
PCAP_PATH = r'C:\Users\USER\Desktop\CNN_anomaly_detection\Image_anomaly_detection\test\test.pcapng'
MODEL_DIR = '../output/cic2018/semi-supervised/model.pth'
# TENSORS_DIR = '../output/cic2018/semi-supervised/tensors' # Base path for npy files
TENSORS_DIR = r'C:\Users\USER\Desktop\CNN_anomaly_detection\Image_anomaly_detection\test'

# Subfolders
TRAIN_BENIGN_DIR = f'{TENSORS_DIR}/train/benign'
TRAIN_OE_DIR = f'{TENSORS_DIR}/train/malicious'
TEST_BENIGN_DIR = f'{TENSORS_DIR}/test/bruteforceXSS1/benign' # CHANGE
TEST_MALICIOUS_DIR = f'{TENSORS_DIR}/test/bruteforceXSS1/malicious' # CHANGE

# Settings
TIME_WINDOW = 60*60 # Time window for calulating OE (seconds)

# --- FlowPic Configuration ---
MAX_PACKET_SIZE = 1500  # Maximum packet size to consider (bytes)
FLOWPIC_DIM = 1500
FLOWPIC_TIME_INTERVAL = 60  # time interval which each FlowPic represents (seconds) 
IMAGE_TYPE = 'binary' # 'binary' is taking > 0 as 1. Others: 'normal'

# --- Aggregation Configuration ---
IMAGE_AGGREGATION = 'summed' # 'summed' is summing FlowPics in the time window
CLIP_SUMMED_COUNTS = True # Set to True to cap pixel values at 255, False to keep raw counts

# # --- Model Configuration ---
# MODEL_NAME = "LeNet5Flowpic_OE"
# INPUT_CHANNELS = 1 # Grayscale
# NUM_CLASSES = 2    # 0: Benign, 1: Malicious (used for OE training)
# DROPOUT_RATE = 0.5

# --- Training Hyperparameters ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10

# Energy-Based OOD Settings (Liu et al. 2020)
T = 1.0
# Paper Eq. 6 Margins:
# We want Benign Energy < -5
# We want Attack Energy > -1
M_IN = -5.0 
M_OUT = -1.0 
OE_LAMBDA = 0.1  # Weight for the energy loss term (usually 0.1)
OOD_THRESHOLD = -3.0  # Energy threshold for classifying as Malicious - must be between M_IN and M_OUT.

# Labels
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1 # This is for testing; during training we use this to identify OE data

# Malicious IPs
ATTACKER_IP = ['18.218.115.60'] # CHANGE
VICTIM_IP =  ['172.31.69.28'] # CHANGE
