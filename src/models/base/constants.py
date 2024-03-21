import torch

# --- TRAINING CONFIGS ---

RANDOM_SEED = 21

IMG_SIZE = 224 #384

BATCH_SIZE = 30 # optimal by formula (gpu_mem - model_size) / (forw_backw_size)

LR = 3e-5

ALPHA2019 = 0.2 # because of nevus distribution

ALPHA_MERG = 0.05

TEST_ALPHA = 0.2

REMOVED_ALPHA = 0.2

BALANCED_ALPHA = 1.

GAMMA = 2

WEIGHT_DECAY = 1e-2

BETA = 0.000000001

N_EPOCHS = 20

T0 = 7

T_MULT = 1

ETA_MIN = 1e-8

LAST_EPOCH = -1

NUM_CLASSES = 2

_mean = torch.Tensor([0.6683, 0.5300, 0.5247])
_std = torch.Tensor([0.2229, 0.2028, 0.2144])

# --- UTILS STRINGS ---

MERGED_TASK = 'merged'

ISIC2019_TASK = '2019'

BALANCED_TASK = 'balanced'

REMOVED_TASK = 'removed'

CUDA_DEVICE = 'cuda'

CPU_DEVICE = 'cpu'

# --- IMAGE FORMATS ---

TRAIN_IMG_FORMAT = '.jpg'

TEST_IMG_FORMAT = '.bmp'

# --- PATHS ---

EPOCH_WEIGHTS_PATH = 'src/models/weights/checkpoints/%(model_name)s/%(epoch)s_%(datetime)s.pth'

MODEL_WEIGHTS_PATH = 'src/models/weights/checkpoints/%(model_name)s/model-%(model_name)s_%(datetime)s.pth'

CORE_PATH = ""

WEIGHTS_PATH = "src/models/weights/checkpoints/%(model_name)s/"

TRAIN_LABELS_PATH2019 = "../isic2019/labels/official/binary_labels2019_2cls.csv"

TRAIN_LABELS_PATH_BALANCED = "../isic2019/labels/official/binary_labels_balanced.csv"

TRAIN_LABELS2019_REMOVED = "../isic2019/labels/official/binary_labels2019_removed.csv" 

TRAIN_LABELS_PATH_MERG = ""

TRAIN_IMG_PATH2019 = "../isic2019_crop/images/official/"

TEST_LABELS_PATH = "../PH2Dataset/binary_labels.csv"

TEST_IMG_PATH = "../PH2Dataset_crop/PH2 Dataset images/"

TRAIN_LABELS_PATH2020 = "../isic2020/labels/binary_labels2020_2cls.csv"

TRAIN_MELANOMA_PATH2020 = "../isic2020/labels/melanoma_labels2020.csv"

TRAIN_IMG_PATH2020 = "../isic2020/images/"

# --- EXCEPTIONS ---

BAD_TASK_ERROR = "Bad dataset task! Please, specify name with 2019/balanced/removed/merged"

