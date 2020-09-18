import streamlit as st


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")

import os

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from food_dataset import FoodDataset, get_calorie

st.markdown("<h1 style='text-align: center; color: black;'>🍟 🍕NUTRITRACK🌭 🍔</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>A Healthier Way For Your LifeStyle</h1>",
            unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Made By Team Boron With ❤️</h1>", unsafe_allow_html=True)

st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
ROOT_DIR = os.getcwd()
FOOD_DIR = os.path.join(ROOT_DIR, "datasets/food/val")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()


# print(ROOT_DIR+"\n"+FOOD_DIR+"\n"+MODEL_DIR)

class Inference_config(Config):
    NAME = 'food'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 12
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10


inf = Inference_config()

model = modellib.MaskRCNN(mode="inference", config=inf, model_dir=MODEL_DIR)

print("\nModel Loaded")

model_path = os.path.join(MODEL_DIR, "model_044.h5")

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_val = FoodDataset()
dataset_val.load_food(FOOD_DIR, "val")
dataset_val.prepare()

x = [i for i in range(1, 192)]
val = st.number_input("Please Enter Image ID BETWEEN 1-190 : ")
st.write("Click ON SUBMIT to get the Predicted results  \n")
if st.button("SUBMIT"):
    st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
    st.write("Image ID Choosen : ", val)

    st.subheader("Your Input Image's Details :  \n")
    st.markdown("<style>hr{border: 2px solid black;}</style>", unsafe_allow_html=True)
    st.write("  \n")
    image_id = int(val)
    image = dataset_val.load_image(image_id)
    st.write("Input Image : ")
    st.image(image=image)

    st.write("  \n")
    results = model.detect([image], verbose=0)
    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'])
    st.write("# Predicted Score :", r['scores'])

    masked_plate_pixels = 1290166.34
    # Average real plate radius
    radius = 6
    real_plate_area = 3.14 * radius * radius
    pixels_per_inch_sq = masked_plate_pixels / real_plate_area
    calories = []
    items = []
    st.title("Calorific Details Are :  \n")
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
    st.write("  \n")
    for i in range(r['masks'].shape[-1]):
        masked_food_pixels = r['masks'][:, :, i].sum()
        class_name = dataset_val.class_names[r['class_ids'][i]]
        real_food_area = masked_food_pixels / pixels_per_inch_sq
        calorie = get_calorie(class_name, real_food_area)
        calories.append(calorie)
        items.append(class_name)
        st.write("## {1} with {0} calories".format(int(calorie), class_name))
    # st.write("Calorific Details Are :  \n")
    # st.write(print("{1} with {0} calories".format(int(calorie), class_name)))
