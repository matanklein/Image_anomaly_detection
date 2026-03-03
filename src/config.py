# Directory structure
SOURCE_PATH = '/mnt/exdisk1/matan/Datasets/CSE-CIC-IDS2018/CSV/benign-15-02.csv' # CHANGE (during preprocessing)
TENSORS_DIR = '/mnt/exdisk1/matan/output/CSE-CIC-IDS2018/semi-supervised/tensors' # Base path for npy files
MODEL_DIR = f'{TENSORS_DIR}/models/model-OOD-DeepSVDD-bruteforceWeb2.pth' # CHANGE (during training/testing)

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

# --- Deep SVDD Configuration ---
LATENT_DIM = 64
SVDD_MARGIN = 50.0  # M_OUT: Distance threshold to push OE samples away

# --- Training Hyperparameters ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20
WEIGHT_DECAY = 1e-5

# --- Sampling / Calibration ---
BALANCED_SAMPLING = True # Oversample minority class (OE) during training
CALIBRATION_STRATEGY = 'youden' # 'youden' or 'f1'
OOD_THRESHOLD = 10.0 # Initial fallback threshold (will be calibrated)
OE_LAMBDA = 1.0

# Labels
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1 # This is for testing; during training we use this to identify OE data
