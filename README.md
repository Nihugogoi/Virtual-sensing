# Virtual-sensing
This project presents the machine learning pipleine to predict the water quality parameters with public datasets. The stratgey is that water temperature data is sued as a primary source to predict two other targets - dissolved oxygen and pH. 

# Chosen model 
Model: Transformer

 
# Important Files (Offline/PC)

Functions_VS.py : All functions are defined in this file and called in the main file where necessary. The functions are related to - (1) model parameters, (2) Transformer model, (3) scale, split and preprocess dataset, (4) model training, (5) model count features, (6) evalutaion of performance metrics, (7) run trained model and visualise

Pretrain_VS_Temp.py : Load dataset and train models

Reconstruct_VS_Temp.py : Implement trained models for data visulation

Convert_ONNX.py : Converts model format from .pth to .onnx 

# Important Files (RPi deployment)
pH_DO_pth.py : inference file in RPi with .pth model

pH_DO_onnx.py : inference file in RPi with .onnx model P.S: The inference test in the edge is done with a small dataset (downsampled to daily data). 

rpi_inference.py : inference time from both .pth model and .onnx model

rpi_metrics.py : performance evaluation of deployed model
 
 
# Important Files (Dataset)
 Trainset : Files/Palma_final.csv 
 
 Testset 1 : Files/Cabrera_final.csv
 
 Testset 2 : Files/Daniel2.csv
 
 RPi test :  Files/Cabrera.csv


 # Original data source
 Trainset: https://apps.socib.es/data-catalog/data-products/buoy_bahiadepalma_physicochemical_parameters
 
 Testset 1: https://emodnet.ec.europa.eu/geonetwork/srv/eng/catalog.search#/metadata/715df37d-0594-4371-ac2d-3336f4fca659
 
 Testset 2: https://www.kaggle.com/datasets/downshift/water-quality-monitoring-dataset
 
 
 
 
