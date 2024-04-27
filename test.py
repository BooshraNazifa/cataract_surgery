from transformers import VivitModel, AutoConfig
import torch

# Load the model
model_name = "google/vivit-b-16x2-kinetics400"
model = VivitModel.from_pretrained(model_name)

# Load the configuration
config = AutoConfig.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode if not training


# ViViT expects input in the form of (batch_size, num_channels, num_frames, height, width)
dummy_input = torch.rand(1, 32, 3, 224, 224)  # Adjust dimensions as necessary

# Forward pass
with torch.no_grad():
    outputs = model(dummy_input, return_dict=True)

# Print output keys to verify structure
print(outputs.keys())