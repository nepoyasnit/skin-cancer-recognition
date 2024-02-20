import numpy as np


RANDOM_SEED = 21

IMG_SIZE = 224

BATCH_SIZE = 277 # optimal by formula (gpu_mem - model_size) / (forw_backw_size)

LR = 3e-05

ALPHA2019 = 0.2 # because of nevus distribution

ALPHA_MERG = 0.2

TEST_ALPHA = 0.2

BALANCED_ALPHA = 1.

GAMMA = 2

WEIGHT_DECAY = 1e-2

BETA = 0.000000001

N_EPOCHS = 20

_mean = np.array([0.6237459654304592, 0.5201169854503829, 0.5039494477029685])

_std = np.array([0.24196317678786788, 0.2233599432947672, 0.23118716487089888])

IMG_FORMAT = '.jpg'

EPOCH_WEIGHTS_PATH = 'weights/checkpoints/%(model_name)s/%(epoch)s_%(datetime)s.pth'

MODEL_WEIGHTS_PATH = 'weights/checkpoints/%(model_name)s/model-%(model_name)s_%(datetime)s.pth'

CORE_PATH = ""

WEIGHTS_PATH = "weights/checkpoints/%(model_name)s/"

TRAIN_LABELS_PATH2019 = "../../isic2019/labels/official/binary_labels2019_2cls.csv"

TRAIN_LABELS_PATH_BALANCED = "../../isic2019/labels/official/binary_labels_balanced.csv"

TRAIN_LABELS_PATH_MERG = ""

TRAIN_IMG_PATH2019 = "../../isic2019/images/official/"

TEST_LABELS_PATH = "../../PH2Dataset/binary_labels.csv"

TEST_IMG_PATH = "../../PH2Dataset/PH2 Dataset images/"

TRAIN_LABELS_PATH2020 = "../../isic2020/labels/binary_labels2020_2cls.csv"

TRAIN_IMG_PATH2020 = "../../isic2020/images/"

