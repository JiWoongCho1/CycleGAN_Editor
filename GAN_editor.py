import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision
import numpy as np
from model_ import Generator



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen_W = Generator(3)
gen_W.load_state_dict(torch.load("Gen_M.pth", device))

gen_M = Generator(3)
gen_M.load_state_dict(torch.load("Gen_W.pth", device))

tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
def image_conversion(image, task, mode):
    if mode == 'man -> girl':
        input = tf(image)
        output = np.array(gen_W(input).detach().permute(1,2,0))
    elif mode == 'girl -> man':
        input = tf(image)
        output = np.array(gen_M(input).detach().permute(1, 2, 0))
    return output


def run():
    st.set_page_config(layout="wide")
    st.title("GAN Editor")
    tab1, tab2, tab3 = st.tabs(["Person", "Background", "building"])
    col2, col3 = st.columns(2)
    with tab1:
        with col2:
            image2 = st.file_uploader("Base Image", ["jpg", "png"])
            if image2:
                image2 = Image.open(image2)
                image2_resized = image2.resize((256, 256))
                st.image(image2_resized, caption="Style Image")
        with col3:
            task = st.selectbox(
                'Which model do you want to use?',
                ("Age", "girl -> man", "man -> girl", 'hair')
            )
            st.write(f"You selected: {task}")
            push_button = st.button("RUN!")
            if push_button and task =='man -> girl':

                output = image_conversion(image2, gen_W, 'man -> girl')
                st.image(output, clamp = True, output_format = 'PNG')

            elif push_button and task =='girl -> man':
                output = image_conversion(image2, gen_M, 'girl -> man')
                st.image(output, clamp=True, output_format='PNG')

if __name__ == "__main__":
    run()