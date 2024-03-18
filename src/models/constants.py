from src.models.base.constants import MERGED_TASK, REMOVED_TASK, BALANCED_TASK, ISIC2019_TASK, CUDA_DEVICE, CPU_DEVICE
                                

# --- TIMM MODELS ---

LEVIT_TIMM = 'levit_256.fb_dist_in1k'

EFFICIENTVIT_TIMM = 'efficientvit_m5.r224_in1k'

EFFICIENTNET_TIMM = 'efficientnet_b4'

# --- BEST MODELS ---
'''
first argument of tuple - name of the .pth file in directory
second argument of tuple - name of the timm model
'''

BEST_MODELS_NAMES = ['efficientnet_b4-f1_0_92', 'efficientnet_b4-weights0.05', 'efficientnet_b4_sens0_9', 
               'efficientvit-f1_0_914', 'efficientvit2019_sens0_825']

BEST_MODELS_TYPES = ['efficientnet_b4', 'efficientnet_b4', 'efficientnet_b4', 'efficientvit_m5.r224_in1k', 'efficientvit_m5.r224_in1k']

# --- PATHS ---

BEST_MODELS_DIR = 'src/models/weights/checkpoints/best/'

TENSORRT_PATH = 'trt_models/float32/'

# --- FORMATS ---

WEIGHTS_FORMAT = '.pth'

TENSORRT_FORMAT = '.ts'
