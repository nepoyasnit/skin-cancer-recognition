import os
import pandas as pd

# --------- ISIC 2016 PREPARATION ---------

train_labels2016 = pd.read_csv('skin-cancer-recognition/ISBI2016_ISIC_Part3_Training_GroundTruth.csv')
test_labels2016 = pd.read_csv('skin-cancer-recognition/ISBI2016_ISIC_Part3_Test_GroundTruth.csv')

downloaded2016_train_data_dir = 'skin-cancer-recognition/ISBI2016_ISIC_Part3_Training_Data/'
downloaded2016_test_data_dir = 'skin-cancer-recognition/ISBI2016_ISIC_Part3_Test_Data/'

for el in train_labels2016.values:
    img = downloaded2016_train_data_dir + el[0] + '.jpg'
    if el[1] == 'benign':
        os.system(f'mv {img} skin-cancer-recognition/data_2016/Train/benign')
    elif el[1] == 'malignant':
        os.system(f'mv {img} skin-cancer-recognition/data_2016/Train/malignant')

for el in test_labels2016.values:
    img = downloaded2016_test_data_dir + el[0] + '.jpg'
    if el[1] == 0.:
        os.system(f'mv {img} skin-cancer-recognition/data_2016/Test/benign')
    elif el[1] == 1.:
        os.system(f'mv {img} skin-cancer-recognition/data_2016/Test/malignant')

os.system(('rm -r skin-cancer-recognition/ISBI2016_ISIC_Part3_Training_Data'))
os.system('rm skin-cancer-recognition/ISBI2016_ISIC_Part3_Training_GroundTruth.csv')
os.system('rm -r skin-cancer-recognition/ISBI2016_ISIC_Part3_Test_Data')
os.system('rm skin-cancer-recognition/ISBI2016_ISIC_Part3_Test_GroundTruth.csv')

# --------- ISIC 2017 PREPARATION ---------
labels_train_2017 = pd.read_csv('skin-cancer-recognition/ISIC-2017_Training_Part3_GroundTruth.csv')
labels_val_2017 = pd.read_csv('skin-cancer-recognition/ISIC-2017_Validation_Part3_GroundTruth.csv')
labels_test_2017 = pd.read_csv('skin-cancer-recognition/ISIC-2017_Test_v2_Part3_GroundTruth.csv')

downloaded2017_train_data_dir = 'skin-cancer-recognition/ISIC-2017_Training_Data/'
downloaded2017_val_data_dir = 'skin-cancer-recognition/ISIC-2017_Validation_Data/'
downloaded2017_test_data_dir = 'skin-cancer-recognition/ISIC-2017_Test_v2_Data/'

train_melanoma_data_dir = 'skin-cancer-recognition/data_2017/Train/melanoma'
train_nevus_data_dir = 'skin-cancer-recognition/data_2017/Train/nevus'
train_sk_data_dir = 'skin-cancer-recognition/data_2017/Train/seborrheic_keratosis'

val_melanoma_data_dir = 'skin-cancer-recognition/data_2017/Val/melanoma'
val_nevus_data_dir = 'skin-cancer-recognition/data_2017/Val/nevus'
val_sk_data_dir = 'skin-cancer-recognition/data_2017/Val/seborrheic_keratosis'

test_melanoma_data_dir = 'skin-cancer-recognition/data_2017/Test/melanoma'
test_nevus_data_dir = 'skin-cancer-recognition/data_2017/Test/nevus'
test_sk_data_dir = 'skin-cancer-recognition/data_2017/Test/seborrheic_keratosis'

for i in range(len(labels_train_2017)):
    filepath = downloaded2017_train_data_dir + labels_train_2017.image_id.iloc[i] + '.jpg'
    if labels_train_2017.melanoma.iloc[i] == 1.0:
        os.system(f'mv {filepath} {train_melanoma_data_dir}')
    elif labels_train_2017.seborrheic_keratosis.iloc[i] == 1.0:
        os.system(f'mv {filepath} {train_sk_data_dir}')
    else:
        os.system(f'mv {filepath} {train_nevus_data_dir}')

for i in range(len(labels_val_2017)):
    filepath = downloaded2017_val_data_dir + labels_val_2017.image_id.iloc[i] + '.jpg'
    if labels_val_2017.melanoma.iloc[i] == 1.0:
        os.system(f'mv {filepath} {val_melanoma_data_dir}')
    elif labels_val_2017.seborrheic_keratosis.iloc[i] == 1.0:
        os.system(f'mv {filepath} {val_sk_data_dir}')
    else:
        os.system(f'mv {filepath} {val_nevus_data_dir}')

for i in range(len(labels_test_2017)):
    filepath = downloaded2017_test_data_dir + labels_test_2017.image_id.iloc[i] + '.jpg'
    if labels_test_2017.melanoma.iloc[i] == 1.0:
        os.system(f'mv {filepath} {test_melanoma_data_dir}')
    elif labels_test_2017.seborrheic_keratosis.iloc[i] == 1.0:
        os.system(f'mv {filepath} {test_sk_data_dir}')
    else:
        os.system(f'mv {filepath} {test_nevus_data_dir}')

os.system('rm -r skin-cancer-recognition/ISIC-2017_Training_Data/')
os.system('rm -r skin-cancer-recognition/ISIC-2017_Test_v2_Data/')
os.system('rm -r skin-cancer-recognition/ISIC-2017_Validation_Data/')

downloaded_lesion_train_data_dir = 'skin-cancer-recognition/ISIC-2017_Training_Part1_GroundTruth/'
downloaded_lesion_val_data_dir = 'skin-cancer-recognition/ISIC-2017_Validation_Part1_GroundTruth/'
downloaded_lesion_test_data_dir = 'skin-cancer-recognition/ISIC-2017_Test_v2_Part1_GroundTruth/'

downloaded_dermo_train_data_dir = 'skin-cancer-recognition/ISIC-2017_Training_Part2_GroundTruth/'
downloaded_dermo_val_data_dir = 'skin-cancer-recognition/ISIC-2017_Validation_Part2_GroundTruth/'
downloaded_dermo_test_data_dir = 'skin-cancer-recognition/ISIC-2017_Test_v2_Part2_GroundTruth/'

lesion_melanoma_train_data_dir = 'skin-cancer-recognition/data_2017/Train_Lesion/melanoma'
lesion_nevus_train_data_dir = 'skin-cancer-recognition/data_2017/Train_Lesion/nevus'
lesion_sk_train_data_dir = 'skin-cancer-recognition/data_2017/Train_Lesion/seborrheic_keratosis'

dermo_melanoma_train_data_dir = 'skin-cancer-recognition/data_2017/Train_Dermo/melanoma'
dermo_nevus_train_data_dir = 'skin-cancer-recognition/data_2017/Train_Dermo/nevus'
dermo_sk_train_data_dir = 'skin-cancer-recognition/data_2017/Train_Dermo/seborrheic_keratosis'

lesion_melanoma_test_data_dir = 'skin-cancer-recognition/data_2017/Test_Lesion/melanoma'
lesion_nevus_test_data_dir = 'skin-cancer-recognition/data_2017/Test_Lesion/nevus'
lesion_sk_test_data_dir = 'skin-cancer-recognition/data_2017/Test_Lesion/seborrheic_keratosis'

dermo_melanoma_test_data_dir = 'skin-cancer-recognition/data_2017/Test_Dermo/melanoma'
dermo_nevus_test_data_dir = 'skin-cancer-recognition/data_2017/Test_Dermo/nevus'
dermo_sk_test_data_dir = 'skin-cancer-recognition/data_2017/Test_Dermo/seborrheic_keratosis'

for i in range(len(labels_train_2017)):
    filepath = downloaded_lesion_train_data_dir + labels_train_2017.image_id.iloc[i] + '_segmentation.png'
    if labels_train_2017.melanoma.iloc[i] == 1.0:
        os.system(f'mv {filepath} {lesion_melanoma_train_data_dir}')
    elif labels_train_2017.seborrheic_keratosis.iloc[i] == 1.0:
        os.system(f'mv {filepath} {lesion_sk_train_data_dir}')
    else:
        os.system(f'mv {filepath} {lesion_nevus_train_data_dir}')

for i in range(len(labels_test_2017)):
    filepath = downloaded_lesion_test_data_dir + labels_test_2017.image_id.iloc[i] + '_segmentation.png'
    if labels_test_2017.melanoma.iloc[i] == 1.0:
        os.system(f'mv {filepath} {lesion_melanoma_test_data_dir}')
    elif labels_test_2017.seborrheic_keratosis.iloc[i] == 1.0:
        os.system(f'mv {filepath} {lesion_sk_test_data_dir}')
    else:
        os.system(f'mv {filepath} {lesion_nevus_test_data_dir}')

