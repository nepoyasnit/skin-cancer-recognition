from base.model import Model
from base.evaluate import evaluate_model
from base.train import _run
from torchsummary import summary


if __name__ == "__main__":
    model = Model('efficientnet_b4',pretrained=True)

    # for i in range(1):
    #     _run(task='merged', model_name='efficientnet_b4', fold=i, model=model)

    test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_model(model, 'efficientnet_b4', 'model-efficientnet_b4_20240304-1515.pth')
    print(f" \
            Test loss: {test_loss}\n \
            Test F1: {test_w_f1}\n \
            Test sensitivity: {test_sens}\n \
            Test specificity: {test_spec}\n \
            Test accuracy: {test_acc}\n \
          ")
