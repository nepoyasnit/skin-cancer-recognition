import torch

from .load_data import load_ph_test_data
from .model import Model
from .loss import FocalLoss
from .constants import TEST_ALPHA, GAMMA, WEIGHTS_PATH, TEST_IMG_PATH, TEST_LABELS_PATH


def evaluate_model(model: Model, model_name: str, model_weights: str, 
                   alpha: float = TEST_ALPHA, gamma: float = GAMMA):
    
    model.load_state_dict(torch.load(WEIGHTS_PATH % {'model_name': model_name} + model_weights))
    test_dataset = load_ph_test_data(TEST_IMG_PATH, TEST_LABELS_PATH)
    
    data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=len(test_dataset))    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    model.to(device)
    
    test_loss, test_w_f1, test_sens, test_spec, test_acc = model.validate_one_epoch(data_loader, 
                                                                                          criterion, 
                                                                                          device)
    return test_loss, test_w_f1, test_sens, test_spec, test_acc
