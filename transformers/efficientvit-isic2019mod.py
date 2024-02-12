import torch

from base.model import Model
from base.evaluate import evaluate_model
from base.train import _run


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model('efficientvit_m5.r224_in1k',pretrained=True)
    model.to(device)

    for i in range(1):
        _run(task='2019', model_name='efficientvit2019', fold=i, model=model)

    # test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_model(model, 'efficientvit2019', 'model-efficientvit_m5_20240208-1838.pth')
    # print(f" \
    #         Test loss: {test_loss}\n \
    #         Test F1: {test_w_f1}\n \
    #         Test sensitivity: {test_sens}\n \
    #         Test specificity: {test_spec}\n \
    #         Test accuracy: {test_acc}\n \
    #       ")
