import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
import time
from PIL import Image
import numpy as np


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")

import os
import plotly
import plotly.graph_objects as go
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from food_dataset import FoodDataset, get_calorie
from weights import check_weights

st.markdown("<h1 style='text-align: center; color: black;'>NutriTrack</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: black;'>A Healthier Way For Your LifeStyle</h1>",
#             unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Made by Team Boron</h1>", unsafe_allow_html=True)

st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center; color: black;'>An intrinsic part of maintaining a healthy lifestyle is "
            "eating right. Nutritionists calculate diets for people based on weight and height and give a specific "
            "amount of calories you are required to eat in a day. But calculating your calorie intake before every "
            "meal can be an annoying and cumbersome task involving a lot of math. NutriTrack will make this job "
            "easier with the click of a button. Try it out now!</h2>", unsafe_allow_html=True)
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

model_path = check_weights()

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_val = FoodDataset()
dataset_val.load_food(FOOD_DIR, "val")
dataset_val.prepare()


with st.spinner('Loading: '):
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)

        my_bar.progress(percent_complete + 1)

    my_bar.empty()

    st.success('Ready to predict!')
    st.subheader("Upload an image:  \n")
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        with st.spinner('Uploading.. Please wait!'):
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.1)

                my_bar.progress(percent_complete + 1)
            my_bar.empty()
            st.success("Uploaded! \n")

    st.markdown("<h2 style='text-align: left; color: black;'>Click on Submit after image is uploaded  \n</h2>",
                unsafe_allow_html=True)

    if st.button("SUBMIT"):

        with st.spinner('Segmenting and predicting calorie details.. Please wait!'):
            my_bar = st.progress(0)

            for percent_complete in range(100):

                time.sleep(0.1)

                my_bar.progress(percent_complete + 1)
            my_bar.empty()
            st.success("Done!  \n")
            st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: black;'>Input image:</h2>",
                        unsafe_allow_html=True)



            st.markdown("<style>hr{border: 2px solid black;}</style>", unsafe_allow_html=True)
            st.write("  \n")

            if uploaded_file is not None:
                image = np.array(Image.open(uploaded_file))
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                st.write("")

                st.markdown("<h2 style=' color: black;'>Classifying... </h2>",
                            unsafe_allow_html=True)
                my_bar = st.progress(0)

                for percent_complete in range(100):
                    time.sleep(0.1)

                    my_bar.progress(percent_complete + 1)
                my_bar.empty()
                st.success("Classified!  \n")

                st.write("  \n")
                results = model.detect([image], verbose=0)
                r = results[0]

                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                            dataset_val.class_names, r['scores'])

                masked_plate_pixels = 1290166.34
                # Average real plate radius
                radius = 6
                real_plate_area = 3.14 * radius * radius
                pixels_per_inch_sq = masked_plate_pixels / real_plate_area
                calories = []
                items = []
                st.title("Calorie Details are:  \n")
                st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
                st.write("  \n")
                for i in range(r['masks'].shape[-1]):
                    masked_food_pixels = r['masks'][:, :, i].sum()
                    class_name = dataset_val.class_names[r['class_ids'][i]]
                    real_food_area = masked_food_pixels / pixels_per_inch_sq
                    calorie = get_calorie(class_name, real_food_area)
                    calories.append(calorie)
                    items.append(class_name)
                    st.write("<h2 style='text-align: left;color: black;'> {1} with {0} calories</h2>".format(int(calorie), class_name),
                                unsafe_allow_html=True)
                fig = go.Figure([go.Bar(x=items, y=calories)])
                fig.update_layout(title='Calorie Graph', autosize=False,
                  width=1400, height=800,xaxis_title="TYPE OF FOOD",
                  yaxis_title="CALORIES",
                  legend_title="Legend Title",
                  font=dict(
                  family="Courier New, monospace",
                  size=18,
                  color="RebeccaPurple"),
                  margin=dict(l=40, r=40, b=40, t=40))
                st.plotly_chart(fig)
                # st.write("Calorific Details Are :  \n")
                # st.write(print("{1} with {0} calories".format(int(calorie), class_name)))