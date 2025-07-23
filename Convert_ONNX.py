from Functions_VS import TransformerModel
import torch
import onnx

# === Model hyperparameters ===
input_size = 1
output_size = 1
nhead = 8
dropout = 0.15
max_seq_len = 1
batch_size = 2
# oxygen model
d_model_oxy = 64
num_encoder_layers_oxy = 8
dim_feedforward_oxy = 128
# ph model
d_model_ph = 32
num_encoder_layers_ph = 6
dim_feedforward_ph= 32
# Load the model
model_oxy = TransformerModel(input_size, d_model_oxy, nhead, num_encoder_layers_oxy,
                             dim_feedforward_oxy, dropout, max_seq_len, output_size)
model_oxy.load_state_dict(torch.load("Models/pretrained_temp2oxy.pth", map_location=torch.device("cpu")))
model_oxy.eval()
#------*********
model_ph = TransformerModel(input_size, d_model_ph, nhead, num_encoder_layers_ph,
                             dim_feedforward_ph, dropout, max_seq_len, output_size)
model_ph.load_state_dict(torch.load("Models/pretrained_temp2ph.pth", map_location=torch.device("cpu")))
model_ph.eval()


# Create a dummy input for ONNX export
input_oxy = torch.randn(2, 1, 1)
input_ph = torch.randn(2, 1, 1)
input_name = "Temperature"

torch.onnx.export(model_oxy, input_oxy, "Models/temp2oxy.onnx",
                  input_names=[input_name], output_names=["Oxygen"],
                  dynamic_axes={input_name: {0: "batch_size", 1: "sequence_length"},
                                "Oxygen": {0: "batch_size"}})

torch.onnx.export(model_ph, input_ph, "Models/temp2ph.onnx",
                  input_names=[input_name], output_names=["PH"],
                  dynamic_axes={input_name: {0: "batch_size", 1: "sequence_length"},
                                "PH": {0: "batch_size"}})











