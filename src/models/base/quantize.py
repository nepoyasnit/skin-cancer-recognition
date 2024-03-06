import torch


def quantize_models(models: list):
    quantized_models = []
    # 'fbgemm' for server, 'qnnpack' for mobile
    backend = 'fbgemm'
    
    for model in models:
        model.to('cpu')

        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        quant_model = torch.quantization.fuse_modules(model.model,
                                                        [['conv', 'bn', 'relu']])
        quant_model = torch.quantization.prepare_qat(quant_model)
        quant_model.eval()

        quant_model = torch.quantization.convert(quant_model)
        # quant_model = torch.ao.quantization.quantize_dynamic(
        #     model,  # the original model
        #     {torch.nn.Linear},  # a set of layers to dynamically quantize
        #     dtype=torch.float16)        
        quantized_models.append(quant_model)
        model.to('cpu')

    return quantized_models
