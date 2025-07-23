from tensorflow.python.keras.utils.version_utils import training
from Functions_VS import train_and_save_model
import pandas as pd


# Load datasets
df = pd.read_csv('Files/Palma_final.csv')

# Filter data for pretraining task: Temperature → Oxygen
df_1 = df.dropna(subset=['Temperature', 'Oxygen'])
X_temp_1 = df_1[['Temperature']]
y_oxy = df_1[['Oxygen']]

df_3 = df.dropna(subset=['Temperature','PH'])
X_temp_3 = df_3[['Temperature']]
y_ph = df_3[['PH']]

# Define filename to save pretrained model
model_oxy = 'Models/pretrained_temp2oxy.pth'
model_ph = 'Models/pretrained_temp2ph.pth'

# Train and save pretrained model
start = time.time()
# model1 = train_and_save_model(X_temp_1,y_oxy, model_oxy)
model3 = train_and_save_model(X_temp_3,y_ph, model_ph)
end = time.time()
training_period = end - start
print(f"Training time for {model_oxy} is {training_period}")



