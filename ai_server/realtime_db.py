import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from model_tools import *
import os
import time
from ultralytics import YOLO
import pandas as pd

cred = credentials.Certificate("/mnt/c/Users/jaeho/Downloads/food-calorie-calculation-app-firebase-adminsdk-tx0fx-83331db9a4.json")
firebase_admin.initialize_app(cred, {'databaseURL':"https://food-calorie-calculation-app-default-rtdb.firebaseio.com/"})


ref = db.reference()


def model_load():
    quantity_model_path = "./quantity.pth"
    yolo_model_path = './yolov8_calories.pt'

    quantity_model, class_to_idx = load_checkpoint(quantity_model_path)
    yolo_model = YOLO(yolo_model_path)
    print("loaded model")
    return quantity_model, yolo_model

def inference(quantity_model, yolo_model, image_path):    
    img = Image.open("images/"+image_path).rotate(180)

    #inference
    quantity = get_quantity(img.copy(), quantity_model, device)
    yolo_result = yolo_model(img)[0]

    # return quantity
    labels = yolo_result.names
    for box in yolo_result.boxes:
        if box.cls == 0:
            continue

        cls = box.cls.cpu()
        food_name = labels[int(cls)]

        kcal = food_df.loc[food_name]["에너지(kcal)"]
        carbohydrate = food_df.loc[food_name]["탄수화물(g)"]
        fat = food_df.loc[food_name]["지방(g)"]
        protein = food_df.loc[food_name]["단백질(g)"]
        sugar = food_df.loc[food_name]["당류(g)"]
        

        
        return food_name, kcal * quantity, carbohydrate * quantity, protein * quantity, fat * quantity, sugar * quantity
    
    return '',0,0,0,0,0

def upload(ref, eventpath, food_name, kcal, carbohydrate, protein, fat, sugar):
    
    # 데이터베이스 참조 가져오기
    epath = eventpath.split("/")
    # 값을 추가할 위치의 참조 가져오기
    users_ref = ref.child(epath[1]).child(epath[2]).child(epath[3])

    # 추가할 데이터 정의
    new_data = {
        "name":food_name,
        "calories":kcal,
        "carbohydrate":carbohydrate,
        "protein":protein,
        "fat":fat,
        "sugar":sugar,
    }

    # 데이터베이스에 데이터 추가
    new_user_ref = users_ref.update(new_data)

    print("data added at realtime db" )


def handle_change(event):
    print(event.data)
    
    print("="*100) 
    
    if time.time() - start < 3:
        print("REALTIME DB CONNTECTED")
        return 0 

    print(event.data, event.path, "\n")
    image_name = event.data['fileName']

    os.system(f"python firebase_storage.py --image_path {image_name} --task download")

    file_downloaded_check = time.time()
    while not os.path.exists(f"images/{image_name}"):
        if time.time() - file_downloaded_check>30:
            print("image download error")
            return -1
    
    food_name, kcal, carbohydrate, protein, fat, sugar = inference(quantity_model, yolo_model, image_name)
    
    upload(ref, event.path, food_name, kcal, carbohydrate, protein, fat, sugar)
    

####################################################################################################


food_df = pd.read_csv('./food_calories_data.csv', index_col="음 식 명")
quantity_model, yolo_model = model_load()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start = time.time()
pre_call = time.time()
    
ref.listen(handle_change)

while True:
    pass
