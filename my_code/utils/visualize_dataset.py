import cv2
import os
import sys
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from dataset.dataset import ImageDataset, DatasetIterator, ImgAugDataset, ImgAugMTDataset, Dataset
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras import losses

def age_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

weights="..\\age_net.hdf5"#"weights/age_pre_tf1.hdf5"#"weights/age_resnet50.hdf5"
model = load_model(weights, custom_objects={"age_mae":age_mae, "age_loss":age_loss})

def visualize_data(dataset):
    for b_x,b_y in dataset:
        for x,y in zip(b_x, b_y):
            # Augment an image
            try:
                pred = model.predict(np.array([x]))
                print("Pred: ", np.argmax(pred))
                print("age: ", np.argmax(y))
                image = cv2.resize(x,(500,500))
                cv2.imshow("img", image)
                #cv2.imshow("img_tf", transformed_image)
                
                cv2.waitKey(0)
            except:
                pass

if __name__=="__main__":
    
    #dataset = ImgAugMTDataset(DatasetIterator(csv_file="data/train.csv", num_classes=82), base_path="../training_caip_contest/")
    dataset = ImageDataset(DatasetIterator(csv_file="../data/val.csv", num_classes=82), base_path="../../training_caip_contest/")
    #dataset.load_from_csv("data/train.csv")
    visualize_data(dataset)
