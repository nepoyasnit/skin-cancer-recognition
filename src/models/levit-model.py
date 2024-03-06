from base.model import Model
from base.evaluate import evaluate_model
from base.train import _run


if __name__ == "__main__":

    model = Model('levit_256.fb_dist_in1k',pretrained=True)

    # for i in range(1):
    #     _run(task='removed', model_name='levit2019', fold=i, model=model)

    test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_model(model, 'levit2019', 'model-levit2019_20240305-1358.pth')
    print(f" \
            Test loss: {test_loss}\n \
            Test F1: {test_w_f1}\n \
            Test sensitivity: {test_sens}\n \
            Test specificity: {test_spec}\n \
            Test accuracy: {test_acc}\n \
          ")

