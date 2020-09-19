import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils


english_lst=['Pudding/custard','Smashed potatoes','Carrots','Spinach','Veal bread cutlet','Oranges','Scallops','Beans','Bread','Yogurt','Pizza','Pasta']
food_diction={'Patate/pure': 2, 'BG': 0, 'Pane': 9, 'Spinaci': 4, 'cotoletta': 5, 'mandarini': 6, 'scaloppine': 7, 'budino': 1, 'carote': 3, 'yogurt': 10, 'pizza': 11, 'fagiolini': 8,'pasta':12}

#calorie_per_square_inch = (calories in food)/(area of plate of dia 12)
calorie_per_sq_inch={'Smashed potatoes':1.4778,'Carrots':0.7256,'Spinach':0.4102,'Veal bread cutlet':4.4247,'Scallops':0.9823,'Beans':0.5486,'Pizza':6.2477,'Pasta':3.5398}
calorie_per_unit={'Pudding/custard':130,'Oranges':62,'Bread':130,'Yogurt':102}

class FoodDataset(utils.Dataset):

    def load_food(self, dataset_dir, mode):

        for i,j in enumerate(english_lst):
        	self.add_class("food", i+1, j)
        
        annotations = json.load(open(os.path.join(dataset_dir, "annotation.json")))

        for a in annotations:
            polygons=annotations[a]
            image_path = os.path.join(dataset_dir,a+".jpg")
            image = skimage.io.imread(image_path)
            h, w = image.shape[:2]

            self.add_image("food", image_id=a+".jpg", path=image_path, width=w, height=h, polygons=polygons)


    def load_mask(self, image_id):

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)

        for _,p in enumerate(info["polygons"]):
            
            p=list(p.values()) 
            rr,cc = skimage.draw.polygon(p[0]['BR'][1::2], p[0]['BR'][::2])  
            mask[rr, cc, i] = 1

        items_names=[''.join(key.keys()) for key in info['polygons']]
        item_ids=list(map(lambda x:food_diction[x],items_names))

        return mask.astype(np.bool), np.array(item_ids, dtype=np.int32)

    def image_reference(self, image_id):

        info = self.image_info[image_id]
        return info["path"]


    def image_pixels(self,image_id):

        info = self.image_info[image_id]
        return info["height"]*info["width"]


def get_calorie(class_name,real_food_area):
    if class_name in calorie_per_unit:
        return calorie_per_unit[class_name]
    else:
        return calorie_per_sq_inch[class_name]*real_food_area