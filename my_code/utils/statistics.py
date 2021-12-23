import cv2
import os
import sys
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from dataset.dataset import ImageDataset, DatasetIterator
from tqdm import tqdm
import numpy as np

    

def compute_median_height_and_width(dataset_iterator:DatasetIterator, base_path=None):
    """
    Args:
        dataset_iterator (DatasetIterator): batch_size=1
    """
    #widths=[]
    #heights=[]
    dict_width = {}
    dict_height = {}
    print_every=10000
    step=1
    for b_x,b_y in tqdm(dataset):
        x = b_x[0]
        path = base_path+x
        img = cv2.imread(path)
        w, h = img.shape[0], img.shape[1]
        dict_width[w] = dict_width[w]+1 if dict_width.get(w) else 1
        dict_height[h] = dict_height[h]+1 if dict_height.get(h) else 1
        if w==h:
            dict_width[w]-=0.5
            dict_height[h]-=0.5

        #widths.append(w)
        #heights.append(h)
        if step%print_every==0:
            print("STEP: ", step)
            wh = []
            values = []
            for w in dict_width.keys():
                #find the corresponding h
                if not dict_height.get(w) is None:
                    h=w
                    wh.append(h)
                    values.append(dict_width[w]+dict_height[h])
            idx = np.argmax(values)
            maxcouple = wh[idx]
            print("Most popular couple of w and h: ", maxcouple)



            #print(np.mean(widths))
            #print(np.mean(heights))        
        step+=1

def cls_nums(dataset):
    df = dataset.df
    print(df.groupby("label").count())



if __name__=="__main__":
    base_path="../training_caip_contest/"
    datasetIterator = DatasetIterator(csv_file="data/train.csv", num_classes=82, batch_size=1)
    compute_median_height_and_width(datasetIterator, base_path=base_path)