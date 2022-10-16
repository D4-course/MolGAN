from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from main_vae import main, Conf
import os
import shutil

app = FastAPI()

img_dir = './exp_results/VAE/2022-10-16_19-18-54/post_test/final/img_dir'

app.mount("/static", StaticFiles(directory=img_dir), name="static")

@app.get("/")
def read_root():
    try:   
        shutil.rmtree(img_dir)
    except FileNotFoundError:
        pass
    
    config = Conf()
    main(config)
    
    all_images = list(os.listdir(img_dir))

    return all_images
