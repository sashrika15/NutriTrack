import os
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from food_dataset import FoodDataset,get_calorie
import numpy as np
from PIL import Image

ROOT_DIR = os.getcwd()
FOOD_DIR = os.path.join(ROOT_DIR, "datasets/food/val")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


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

image = np.array(Image.open("/Users/sashrikasurya/Documents/NutriTrack/datasets/food/val/20151127_122032.jpg"))

results = model.detect([image], verbose=0)
r = results[0]

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'])


############# CALORIE CALCULATION #########################
#Basic intuition is that the ratio of real plate to image plate area = ratio of image food to real food area
#masked_plate_pixels has been calculated with annotation tools
masked_plate_pixels=1290166.34 
#Average real plate radius
radius = 6
real_plate_area=3.14*radius*radius
pixels_per_inch_sq=masked_plate_pixels/real_plate_area
calories=[]
items=[]
for i in range(r['masks'].shape[-1]):
  masked_food_pixels=r['masks'][:,:,i].sum()
  class_name=dataset_val.class_names[r['class_ids'][i]]
  real_food_area=masked_food_pixels/pixels_per_inch_sq
  calorie=get_calorie(class_name,real_food_area)
  calories.append(calorie)
  items.append(class_name)
  print("{1} with {0} calories".format(int(calorie),class_name))