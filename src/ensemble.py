import torch

from base.model import Model
from base.evaluate import evaluate_ensemble


if __name__ == "__main__":
    efficientnet = Model('efficientnet_b4',pretrained=True)
    efficientnet.load_state_dict(torch.load('weights/checkpoints/best/efficientnet_b4-f1_0_92.pth'))

    efficientnet2 = Model('efficientnet_b4',pretrained=True)
    efficientnet2.load_state_dict(torch.load('weights/checkpoints/best/efficientnet_b4-weights0.05.pth'))

    efficientnet3 = Model('efficientnet_b4',pretrained=True)
    efficientnet3.load_state_dict(torch.load('weights/checkpoints/best/model-efficientnet_b4_20240304-1515.pth'))

    efficientvit = Model('efficientvit_m5.r224_in1k',pretrained=True)
    efficientvit.load_state_dict(torch.load('weights/checkpoints/best/efficientvit-f1_0_914.pth'))

    efficientvit2 = Model('efficientvit_m5.r224_in1k',pretrained=True)
    efficientvit2.load_state_dict(torch.load('weights/checkpoints/best/efficientvit2019_sens0_825.pth'))


    models = [efficientnet, efficientnet2, efficientnet3, efficientvit, efficientvit2]

    test_w_f1, test_sens, test_spec, test_acc = evaluate_ensemble(models)
    print(f" \
            Test F1: {test_w_f1}\n \
            Test sensitivity: {test_sens}\n \
            Test specificity: {test_spec}\n \
            Test accuracy: {test_acc}\n \
          ")

