from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

caltech_dir = "./DLclass/certification/10"
image_w = 64
image_h = 64

CG=0
GC=0
GIM=0
HF=0
HIN=0
HP=0
LIN=0
Sum_i = np.zeros((1,7))
GD = ''

answer=0

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
model = load_model('./model/multi_img_classification.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0


for i in prediction:
    pre_ans = i.argmax()
#    print(i)
 #   print(pre_ans)
    Sum_i = Sum_i + i
    cnt += 1

my_answer = np.argmax(Sum_i)
if   my_answer == 0: GD = "Chronic_Gastritis"
elif my_answer == 1: GD = "Gastric_cancer"
elif my_answer == 2: GD = "Gastric_Intestinal_Metaplasia"
elif my_answer == 3: GD = "Healthy_Fundus"
elif my_answer == 4: GD = "High_Grade_Intraepithelial_Neoplasia"
elif my_answer == 5: GD = "Healthy_Pylorus"
else: GD = "Low_Grade_Intraepithelial_Neoplasia"
print("이미지는 "+GD+"로 추정됩니다.")