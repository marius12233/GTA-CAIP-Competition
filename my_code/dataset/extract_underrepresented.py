import sys
import os
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from dataset import Dataset
import numpy as np
import shutil

source_img = "..\\training_caip_contest"
dataset = Dataset(csv_file="data/train.csv", num_classes=101, to_categorical=False)
#classes_counts = dataset.get_classes_counts()
min_occ_class = 1000
labels = dataset.data.y

classes_counts = dataset.get_classes_counts()
for class_ in classes_counts.keys():
    count = classes_counts[class_]
    idxs = np.where(labels==class_)[0]
    filtered_idxs = []
    if count<min_occ_class:
        #choise max_occ_class indexes from class class_
        sampled_idxs = idxs.tolist()
        filtered_idxs+=sampled_idxs
    x = dataset.data.x[filtered_idxs]
    base = os.path.join("C:\\Users\\mario\\underrepresenteted_age\\",str(class_))
    
    os.mkdir(base)
    os.chmod(base, 0o777)
    for img_name in x:
        src=os.path.join(source_img, img_name)
        dst=os.path.join(base, img_name)
        print(src, dst)
        shutil.copyfile(src, dst)



    print(class_)








