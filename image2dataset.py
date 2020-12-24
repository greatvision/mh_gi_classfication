from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split

caltech_dir = "./DLclass//train"
categories = ["CG", "GC", "GIM", "HF", "HIN", "HP", "LIN"]
nb_classes = len(categories)

image_w = 64
image_h = 64

pixels = image_h * image_w * 1

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")#그레이스케일("L")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)
xy = (X_train, X_test, y_train, y_test)
np.save("./img_data_3.npy", xy)

print("ok", len(y))