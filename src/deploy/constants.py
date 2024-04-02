TENSORRT_PATH = 'src/deploy/saved_models/float16/'

TENSORRT_FORMAT = '.ts'

TS_MODELS_NUM = 3 # amount of models, which we can convert to TensorRT (EfficientNet's)

TRT_MODELS_NAMES = ['model-efficientnet_b4_crop384', 'model-efficientnet_b4_sens0.05', 'model-efficientnet_b4', 
               'model-efficientvit_1e-4', 'model-efficientvit']

TRT_MODELS_TYPES = ['efficientnet_b4', 'efficientnet_b4', 'efficientnet_b4', 'efficientvit_m5.r224_in1k', \
                    'efficientvit_m5.r224_in1k']

MODEL_INPUT_DIM = (1, 3, 224, 224)
