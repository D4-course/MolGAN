import streamlit as st
import json
import requests

BASE_URL = "http://localhost:8000/"
IMG_BASE_URL = BASE_URL+"static/"

def home_page():
    st.caption("> Generates Drug Like Molecules using GANs")
    generate = st.button(label="Click here to generate molecules")
    if generate: 
        st.markdown("""---""")
        placeHolder = st.empty()
        with st.spinner("Generating Molecules"):
            try:
                resp = requests.get(BASE_URL)
                imgs = resp.json()
                with placeHolder.container():
                    num_imgs = len(imgs)
                    
                    for img1, img2 in zip(imgs[:num_imgs//2], imgs[num_imgs//2:]):

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(IMG_BASE_URL+img1, caption=img1)
                        # with col2:
                        #     st.write(' ')
                        with col2:
                            st.image(IMG_BASE_URL+img2, caption=img2)
            except:
                placeHolder.error("Error in retrieving data")

st.set_page_config(
    page_title="MOL-GAN"
)

st.title("MOL-GAN")
home_page()