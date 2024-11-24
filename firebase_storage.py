import firebase_admin
from firebase_admin import credentials, storage
import argparse

from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from ultralytics import YOLO
import cv2


def download_image(image_name):
    image_path = "images/"+image_name
    
    local_image_path = f"images/{image_name}"

    blob = bucket.blob(image_path)
    blob.download_to_filename(local_image_path)

    print(f"Image downloaded to {local_image_path}")

def upload_image(image_path, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    with open("images/"+f"{image_path}", 'rb') as image_file:
        blob.upload_from_file(image_file)

    print(f'Image uploaded to {destination_blob_name}')


############################################################################################################################################################
#parse

parser = argparse.ArgumentParser(description='firebase_image download')    
parser.add_argument('--image_path')
parser.add_argument('--task')
args = parser.parse_args()    # 4. 인수를 분석

image_name = args.image_path
task = args.task


############################################################################################################################################################
# login
cred = credentials.Certificate("/mnt/c/Users/jaeho/Downloads/food-calorie-calculation-app-firebase-adminsdk-tx0fx-83331db9a4.json")

# Firebase 프로젝트의 서비스 계정 키 파일 경로
firebase_admin.initialize_app(cred, {
    'storageBucket': 'food-calorie-calculation-app.appspot.com'
})

bucket = storage.bucket()


# execute

if task == "download":
    download_image(image_name)
elif task == "upload":
    upload_image(image_name, f'images/{image_name}')
else:
    print("invalid argument")