#! /bin/bash
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv

unzip ISBI2016_ISIC_Part3_Training_Data.zip
rm ISBI2016_ISIC_Part3_Training_Data.zip

unzip ISBI2016_ISIC_Part3_Test_Data.zip
rm ISBI2016_ISIC_Part3_Test_Data.zip

mkdir data_2016/
mkdir data_2016/Train/
mkdir data_2016/Train/benign/
mkdir data_2016/Train/malignant/

mkdir data_2016/Test/
mkdir data_2016/Test/benign/
mkdir data_2016/Test/malignant/

mv ISBI2016_ISIC_Part3_Training_Data/ISIC_0000000.jpg data_2016/Train/benign
mv ISBI2016_ISIC_Part3_Test_Data/ISIC_0000003.jpg data_2016/Test/benign

mkdir image_paths

touch image_paths/train.csv
touch image_paths/train_oversample.csv
touch image_paths/val.csv
touch image_paths/test.csv
mkdir logs/checkpoints

mkdir results
mkdir data_labels

mv ISBI2016_ISIC_Part3_Test_GroundTruth.csv data_labels/ISBI2016_ISIC_Part3_Test_GroundTruth.csv
mv ISBI2016_ISIC_Part3_Training_GroundTruth.csv data_labels/ISBI2016_ISIC_Part3_Training_GroundTruth.csv
mv ISIC-2017_Test_v2_Part3_GroundTruth.csv data_labels/ISIC-2017_Test_v2_Part3_GroundTruth.csv
mv ISIC-2017_Training_Part3_GroundTruth.csv data_labels/ISIC-2017_Training_Part3_GroundTruth.csv
mv ISIC-2017_Validation_Part3_GroundTruth.csv data_labels/ISIC-2017_Validation_Part3_GroundTruth.csv
