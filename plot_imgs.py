import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from textwrap import wrap
import argparse as ap

plt.rcParams["font.family"] = "serif"

def label_imgs(dir, display=False):
    df = pd.read_csv(os.path.join(dir, "desc.csv"))
    # print(df)

    img1, prompt1 = df.loc[12, "image"], df.loc[12, "prompt"]
    img2, prompt2 = df.loc[1, "image"], df.loc[1, "prompt"]
    print(prompt2)
    prompt1 = prompt1 + ", by yukaryote"
    prompt2 = prompt2 + ", by yukaryote"

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        img1_plot = plt.imshow(cv.cvtColor(cv.imread(img1), cv.COLOR_BGR2RGB))
        ax.set_title("\n".join(wrap(prompt1, 50)), fontsize=10)
        plt.axis("off")
        ax = fig.add_subplot(1, 2, 2)
        img2_plot = plt.imshow(cv.cvtColor(cv.imread(img2), cv.COLOR_BGR2RGB))
        ax.set_title("\n".join(wrap(prompt2, 50)), fontsize=10)
        plt.axis("off")
        plt.show()

if __name__== "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--path", help="path to dataset directory")

    args = parser.parse_args()

    label_imgs(args.path)