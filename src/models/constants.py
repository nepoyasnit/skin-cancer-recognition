from src.models.base.constants import MERGED_TASK, REMOVED_TASK, BALANCED_TASK, ISIC2019_TASK, CUDA_DEVICE, CPU_DEVICE, \
                                    IMG_SIZE
                                

# --- TIMM MODELS ---

LEVIT_TIMM = 'levit_256.fb_dist_in1k'

EFFICIENTVIT_TIMM = 'efficientvit_m5.r224_in1k'

EFFICIENTNET_TIMM = 'efficientnet_b4'

ATTENTION_MODEL = 'model-efficientvit'

# --- BEST MODELS ---
'''
first argument of tuple - name of the .pth file in directory
second argument of tuple - name of the timm model
'''

BEST_MODELS_NAMES = ['model-efficientnet_b4_crop384', 'model-efficientnet_b4_sens0.05', 'model-efficientnet_b4', 
               'model-efficientvit_1e-4', 'model-efficientvit']

BEST_MODELS_TYPES = ['efficientnet_b4', 'efficientnet_b4', 'efficientnet_b4', 'efficientvit_m5.r224_in1k', 'efficientvit_m5.r224_in1k']

# --- PATHS ---

BEST_MODELS_DIR = 'src/models/weights/checkpoints/best/'

# --- FORMATS ---

WEIGHTS_FORMAT = '.pth'

