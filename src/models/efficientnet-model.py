from src.models.base.model import Model
from src.models.base.evaluate import evaluate_model
from src.models.base.train import _run
from src.models.constants import MERGED_TASK, EFFICIENTNET_TIMM


if __name__ == "__main__":
    model = Model(EFFICIENTNET_TIMM, pretrained=True)

    for i in range(1):
        _run(task=MERGED_TASK, model_name=EFFICIENTNET_TIMM, fold=i, model=model)

    # test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_model(model, EFFICIENTNET_TIMM, 'model-efficientnet_b4_20240304-1515.pth')
    # print(f" \
    #         Test loss: {test_loss}\n \
    #         Test F1: {test_w_f1}\n \
    #         Test sensitivity: {test_sens}\n \
    #         Test specificity: {test_spec}\n \
    #         Test accuracy: {test_acc}\n \
    #       ")
