import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import your model
from Network.Network import CIFAR10Model, resnet34, resnext34, densenet, mobilenetv2  

# Load your trained model
model = CIFAR10Model(InputSize=3 * 32 * 32, OutputSize=10) # Provide the actual input and output sizes
Checkpoint = torch.load('../../Checkpoints/customsmallnet_t1/47model.ckpt')
model.load_state_dict(Checkpoint["model_state_dict"])  # Load your trained model weights

# Print Model layers
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Define a target layer for Grad-CAM (replace 'convlayer3' with the layer you want)
target_layers = [model.convlayer3[-1]]

# Load an example image from the CIFAR-10 dataset (replace 'path_to_image' with the actual path)
image_path = "../CIFAR10/Test/10.png"
image = Image.open(image_path).convert('RGB')
print(image.size)
# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match the input size of the model
    transforms.ToTensor(),
])

input_image = preprocess(image).unsqueeze(0)  # Add batch dimension
print(input_image.shape)
# Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

# Forward pass
logits = model(input_image)
target_class = torch.argmax(logits)

print(target_class)
# Generate CAM
cam_image = cam(input_image, target_class)

# Visualize the result
visualization = show_cam_on_image(input_image.squeeze(), cam_image.squeeze())
visualization.show()
