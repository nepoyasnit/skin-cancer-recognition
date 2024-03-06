import torch
from torchsummary import summary

from base.model import Model


def get_best_model():
    device = torch.device('cuda')

    efficientnet = Model('efficientnet_b4',pretrained=True)
    efficientnet.load_state_dict(torch.load('weights/checkpoints/best/efficientnet_b4-f1_0_92.pth', map_location=device))

    efficientnet2 = Model('efficientnet_b4',pretrained=True)
    efficientnet2.load_state_dict(torch.load('weights/checkpoints/best/efficientnet_b4-weights0.05.pth', map_location=device))

    efficientnet3 = Model('efficientnet_b4',pretrained=True)
    efficientnet3.load_state_dict(torch.load('weights/checkpoints/best/model-efficientnet_b4_20240304-1515.pth', map_location=device))

    efficientvit = Model('efficientvit_m5.r224_in1k',pretrained=True)
    efficientvit.load_state_dict(torch.load('weights/checkpoints/best/efficientvit-f1_0_914.pth', map_location=device))

    efficientvit2 = Model('efficientvit_m5.r224_in1k',pretrained=True)
    efficientvit2.load_state_dict(torch.load('weights/checkpoints/best/efficientvit2019_sens0_825.pth', map_location=device))

    # levit = Model('levit_256.fb_dist_in1k',pretrained=True)
    # levit.load_state_dict(torch.load('weights/checkpoints/best/model-levit2019_20240305-1358.pth', map_location=device))

    models = [efficientnet, efficientnet2, efficientnet3, efficientvit, efficientvit2]

    return models