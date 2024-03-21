from src.models.base.model import Model
from src.models.base.evaluate import evaluate_model
from src.models.base.train import _run
from src.models.constants import EFFICIENTVIT_TIMM, MERGED_TASK


if __name__ == "__main__":
    model = Model(EFFICIENTVIT_TIMM, pretrained=True)

    for i in range(1):
        _run(task=MERGED_TASK, model_name=EFFICIENTVIT_TIMM, fold=i, model=model)

    # test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_model(model, 'efficientvit2019', 'model-efficientvit2019_20240229-1815.pth')
    # print(f" \
    #         Test loss: {test_loss}\n \
    #         Test F1: {test_w_f1}\n \
    #         Test sensitivity: {test_sens}\n \
    #         Test specificity: {test_spec}\n \
    #         Test accuracy: {test_acc}\n \
    #       ")
