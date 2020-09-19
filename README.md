# NutriTrack

### Motivation
An intrinsic part of maintaining a healthy lifestyle is eating right. Nutritionists calculate diets for people based on weight and height and give a specific amount of calories you are required to eat in a day. But calculating your calorie intake before every meal can be an annoying and cumbersome task involving a lot of math. NutriTrack will make this job easier with the click of a button. Just give an image of your plate as input to the application, and it will calculate the amount of calories you are eating in that meal.

### Model Details
* We have used [matterport's](https://github.com/matterport/Mask_RCNN) implementation of Mask R-CNN for Object Detection and Segmentation. Check out the paper [here](https://arxiv.org/abs/1703.06870)!
* UNIMIB2016 food dataset has been used for training with some classes merged due to small size of the dataset. 
* Our model is currently still training, we have used the best available weights for this implementation.

### Installation
```
git clone https://github.com/sashrika15/NutriTrack.git 
```

### To run
Change you directory to NutriTrack, then run the following command

```
streamlit run app.py
```


### Problems
We ran into problems during calorie calculation and where unable to calculate depth of the food (through which we would have calculated volume, then mass of food), hence we had to look for other methods for calorie calculation. 
Right now, our model calculates calories with the help of two main parameters:
* Calories per square inch, which has been calculated as the ratio of calories in one serving of the food to the area of plate
* Real food area, which has been calculated by equating the ratios of masked_food_pixels/masked_plate_pixels to real_food_area/real_plate_area
* Calories = Calories_per_sq_inch * real_food_area

### References
Thanks to [matterport](https://github.com/matterport/Mask_RCNN) and [binayakpokhrel](https://github.com/binayakpokhrel) for their respective implementations.