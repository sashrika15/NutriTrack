import gdown
import os

ROOT_DIR = os.getcwd()

def check_weights():

    if os.path.exists(ROOT_DIR + "/logs/model.h5"):
        print("Found weights")
    else:
        print("Downloading weights")
        if not os.path.exists(ROOT_DIR+"/logs/"):
            os.mkdir(ROOT_DIR+"/logs/")
    
        url = "https://drive.google.com/uc?export=download&id=1KpfckgLEpU9aAsZSJEvPCnraYwvuY0Xz"
        gdown.download(url, ROOT_DIR+"/logs/model.h5", quiet=False)
    return ROOT_DIR+"/logs/model.h5"
    
