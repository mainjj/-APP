from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2

def load_checkpoint(filepath, map_location='cpu'):
    checkpoint = torch.load(filepath)
    model = checkpoint['model_ft']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint['class_to_idx']

def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    image = preprocess(image)
    return image

def get_quantity(img, model, device):
    topk=5
    img = process_image(img).unsqueeze(0).to(device)

    model.eval()

    logits = model.forward(img)

    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    probs, classes = (e.data.numpy().squeeze().tolist() for e in topk)

    return (classes[np.argmax(probs)]+1) * 0.25


