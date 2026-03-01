# Directory structure
SOURCE_PATH = '../data/cic2018/bruteforce-xss-1-22_02_2018.pcap' # CHANGE
TENSORS_DIR = '../output/cic2018/semi-supervised/tensors' # Base path for npy files
MODEL_DIR = f'{TENSORS_DIR}/model.pth'

# Subfolders
TRAIN_DIR = f'{TENSORS_DIR}/train'
TEST_BENIGN_DIR = f'{TENSORS_DIR}/test/bruteforceWeb2/benign' # CHANGE (during test)
TEST_MALICIOUS_DIR = f'{TENSORS_DIR}/test/bruteforceWeb2/malicious' # CHANGE (during test)

# Malicious IPs
ATTACKER_IP = ['18.218.115.60'] # CHANGE (during preprocessing)
VICTIM_IP =  ['172.31.69.28'] # CHANGE (during preprocessing)

# Settings

# --- FlowPic Configuration ---
MAX_PACKET_SIZE = 1500  # Maximum packet size to consider (bytes)
FLOWPIC_DIM = 1500
FLOWPIC_TIME_INTERVAL = 60  # time interval which each FlowPic represents (seconds) 
IMAGE_TYPE = 'binary' # 'binary' is taking > 0 as 1 when creating FlowPic. 'limited_count' is capping counts at 255 but keeping raw counts. else 'normal' is keeping raw counts without capping.

# --- Aggregation Configuration ---
IMAGE_AGGREGATION = 'summed' # 'summed' is summing FlowPics in the time window
CLIP_SUMMED_COUNTS = False # Set to True to cap pixel values at 255, False to keep raw counts

# # --- Model Configuration ---
MODEL_NAME = "LeNet5Flowpic_OE"
INPUT_CHANNELS = 1 # Grayscale
NUM_CLASSES = 2    # 0: Benign, 1: Malicious (used for OE training)
DROPOUT_RATE = 0.5

# --- Training Hyperparameters ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20

# --- Sampling / Calibration ---
BALANCED_SAMPLING = True   # Oversample minority class (OE) during training
CALIBRATION_STRATEGY = 'youden'  # 'youden' or 'f1'

# Energy-Based OOD Settings (Liu et al. 2020)
T = 1.0
# Paper Eq. 6 Margins:
# We want Benign Energy < -5
# We want Attack Energy > -1
M_IN = -5.0 
M_OUT = -1.0 
OE_LAMBDA = 1.0  # Weight for OE margin term (higher to avoid benign-only collapse)
OOD_THRESHOLD = -3.0  # Energy threshold for classifying as Malicious - must be between M_IN and M_OUT.

# Labels
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1 # This is for testing; during training we use this to identify OE data
