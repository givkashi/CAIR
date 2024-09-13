import os
import cv2
import torch
import numpy as np
import gradio as gr
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from omegaconf import OmegaConf
import yaml

def load_model(checkpoint_path, device='cpu'):
    """Load the model from a given checkpoint."""

    train_config_path = os.path.join(checkpoint_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(checkpoint_path,
                                   'models',
                                   'last.ckpt')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    return model


def preprocess(image, mask):
    """Preprocess the input image and mask before feeding them into the model."""
    image = np.array(image)
    mask = np.array(mask)
    image = image.astype('float32') / 255

    mask = mask.astype('float32') / 255

    # Convert to tensors and batch them
    batch = {
        'input_image': torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float(),
        'mask': torch.tensor(mask).permute(2, 0, 1).unsqueeze(0).float()
    }

    return batch


def postprocess(output):
    """Postprocess the output to convert it into an image."""
    cur_res = output[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def predict(image, mask, checkpoint_path):
    """Run prediction using the model."""
    device = torch.device('cpu')  # Modify this if using GPU
    model = load_model(checkpoint_path, device)

    batch = preprocess(image, mask)
    batch = move_to_device(batch, device)

    with torch.no_grad():
        output = model(batch)  # Replace 'output' with the actual key from your model
        output = output['predicted_image']

    result = postprocess(output)
    return result


# Define Gradio interface
def gradio_interface(image, mask):
    checkpoint_path = 'F:\CAIR-main\checkpoint'  # Provide the correct checkpoint path
    return predict(image, mask, checkpoint_path)


# Gradio app layout
inputs = [
    gr.Image(type="pil", label="Input Image"),
    gr.Image(type="pil", label="Input Mask")
]

outputs = gr.Image(type="numpy", label="Output Image")

title = "Supervised Deep Learning for Content-Aware Image Retargeting with Fourier Convolutions"
description = "Gradio demo for Image Retargeting. To use it, simply upload your image and mask, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://link.springer.com/article/10.1007/s11042-024-18876-8' target='_blank'>Article</a> | <a href='https://github.com/givkashi/CAIR' target='_blank'>Github Repo</a></p>"
examples = [
    ['test_images/input/im1.png','test_images/input/im1_mask.png'],
    ['test_images/input/im3.png','test_images/input/im3_mask.png'],
]

# Launch Gradio Interface
gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs, title=title, description=description, article=article, examples=examples).launch()