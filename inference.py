import random
import os
import sys

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.config import Config
from food_dataset import FoodDataset,get_calorie
import matplotlib.pyplot as plt


ROOT_DIR = os.getcwd()
FOOD_DIR = os.path.join(ROOT_DIR, "datasets/food/val")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#print(ROOT_DIR+"\n"+FOOD_DIR+"\n"+MODEL_DIR)

class Inference_config(Config):
    NAME='food'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 12
    RPN_ANCHOR_SCALES = (4,8,16, 32,64)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10

inf = Inference_config()

model = modellib.MaskRCNN(mode="inference", config=inf, model_dir=MODEL_DIR)

print("\nModel Loaded")

model_path = os.path.join(MODEL_DIR,"model_044.h5")

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_val = FoodDataset()
dataset_val.load_food(FOOD_DIR,"val")
dataset_val.prepare()


image_id = 5
image = dataset_val.load_image(image_id)
results = model.detect([image], verbose=0)
r = results[0]

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'])

