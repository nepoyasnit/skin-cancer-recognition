#! /bin/bash

wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part2_GroundTruth.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part3_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part2_GroundTruth.zip
wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part2_GroundTruth.zip

unzip ISIC-2017_Training_Data.zip
rm ISIC-2017_Training_Data.zip

unzip ISIC-2017_Test_v2_Data.zip
unzip ISIC-2017_Training_Data.zip
unzip ISIC-2017_Test_v2_Part1_GroundTruth.zip
unzip ISIC-2017_Training_Part1_GroundTruth.zip
unzip ISIC-2017_Training_Part2_GroundTruth.zip
unzip ISIC-2017_Validation_Data.zip
unzip ISIC-2017_Validation_Part1_GroundTruth.zip
unzip ISIC-2017_Validation_Part2_GroundTruth.zip
unzip ISIC-2017_Test_v2_Part2_GroundTruth.zip

rm ISIC-2017_Test_v2_Data.zip
rm ISIC-2017_Training_Data.zip
rm ISIC-2017_Test_v2_Part1_GroundTruth.zip
rm ISIC-2017_Training_Part1_GroundTruth.zip
rm ISIC-2017_Training_Part2_GroundTruth.zip
rm ISIC-2017_Validation_Data.zip
rm ISIC-2017_Test_v2_Part2_GroundTruth.zip
rm ISIC-2017_Validation_Part1_GroundTruth.zip
rm ISIC-2017_Validation_Part2_GroundTruth.zip

mkdir data_2017
mkdir data_2017/Train
mkdir data_2017/Val
mkdir data_2017/Test
mkdir data_2017/Train_Lesion
mkdir data_2017/Train_Dermo
mkdir data_2017/Test_Lesion
mkdir data_2017/Test_Dermo

mkdir data_2017/Train/melanoma
mkdir data_2017/Train/nevus
mkdir data_2017/Train/seborrheic_keratosis

mkdir data_2017/Val/melanoma
mkdir data_2017/Val/nevus
mkdir data_2017/Val/seborrheic_keratosis

mkdir data_2017/Test/melanoma
mkdir data_2017/Test/nevus
mkdir data_2017/Test/seborrheic_keratosis

mkdir data_2017/Train_Lesion/melanoma
mkdir data_2017/Train_Lesion/nevus
mkdir data_2017/Train_Lesion/seborrheic_keratosis

mkdir data_2017/Train_Dermo/melanoma
mkdir data_2017/Train_Dermo/nevus
mkdir data_2017/Train_Dermo/seborrheic_keratosis

mkdir data_2017/Test_Lesion/melanoma
mkdir data_2017/Test_Lesion/nevus
mkdir data_2017/Test_Lesion/seborrheic_keratosis

mkdir data_2017/Test_Dermo/melanoma
mkdir data_2017/Test_Dermo/nevus
mkdir data_2017/Test_Dermo/seborrheic_keratosis

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
