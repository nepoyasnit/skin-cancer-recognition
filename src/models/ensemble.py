import torch
from torchsummary import summary

from src.models.best_model import get_best_model
from src.models.base.model import Model
from src.models.base.evaluate import evaluate_ensemble


if __name__ == "__main__":
    models, names = get_best_model()

    test_loss, test_w_f1, test_sens, test_spec, test_acc = evaluate_ensemble(models)
    print(f" \
            Test loss: {test_loss}\n \
            Test F1: {test_w_f1}\n \
            Test sensitivity: {test_sens}\n \
            Test specificity: {test_spec}\n \
            Test accuracy: {test_acc}\n \
          ")

