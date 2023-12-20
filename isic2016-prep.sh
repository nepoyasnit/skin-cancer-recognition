#! /bin/bash
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv

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

touch train.csv
touch train_oversample.csv
touch val.csv
touch test.csv
mkdir logs/checkpoints
