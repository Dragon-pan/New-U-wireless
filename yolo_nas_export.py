# Load model with pretrained weights
import torch
import onnx
from super_gradients.training import models
from super_gradients.common.object_names import Models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_res = [640, 640]
batch_size = 1

# Get model
model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").to(device)

# Prepare model for conversion
# Input size is in format of [Batch x Channels x Width x Height] where 640 is the standard COCO dataset dimensions
model.eval()
model.prep_model_for_conversion(input_size=[batch_size, 3, input_res[0], input_res[1]])
    
# Create dummy_input
dummy_input = torch.randn(batch_size, 3, input_res[0], input_res[1], device=device)

# Convert model to onnx
model_name = "yolo_nas_s_" + str(input_res[0]) + "x" + str(input_res[1]) + ".onnx"
torch.onnx.export(model, dummy_input, model_name, verbose=False)

model = onnx.load(model_name)

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(50*"-")
print(onnx.helper.printable_graph(model.graph))