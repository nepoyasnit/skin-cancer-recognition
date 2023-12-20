# Skin cancer recognition neural network

## TO START:

- chmod +x isic2016-prep.sh
- ./isic2016-prep.sh
- chmod +x isic2017-prep.sh
- ./isic2017-prep.sh

## TO CHECK METRICS AFTER TRAINING:

tensorboard --logdir=logs

## RESULTS: 
val result: accuracy 87.33%

mean precision 81.84% mean recall 75.83% 
precision for mel 73.91% recall for mel 56.67%

AP 0.6910 AUC 0.8800 optimal AUC: 0.8861