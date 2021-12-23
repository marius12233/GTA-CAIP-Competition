import cv2
import os
import sys
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from dataset.dataset import DatasetIterator, ImageDataset
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras import losses
import albumentations as A
import numpy as np
import cv2
import random
from pychubby.actions import Chubbify, Multiple, Pipeline, Smile
from pychubby.detect import LandmarkFace

# Declare an augmentation pipeline

transform = A.Compose([

    A.HorizontalFlip(p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
    A.Blur(blur_limit=5, always_apply=False, p=0.3),
    A.RandomShadow(p=0.2),
    A.RGBShift(p=0.2),
    A.RandomFog(alpha_coef=0.2, p=0.2),
    A.ToGray(p=0.5),
    A.ShiftScaleRotate(p=0.8),
    A.RandomSunFlare(p=0.2, src_radius=80),
    A.CenterCrop(p=0.2, height=192, width=192),
    A.GaussNoise(p=0.5)
])
    #A.RandomCrop(width=128, height=128, p=0.2),
    #A.ColorJitter(p=1),
    #A.Downscale(p=0.2),
    #A.ISONoise(p=1), 
def augment(image, bgr=True, img_size=(224,224), p=0.6):
    if random.random()<p:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        if bgr:
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        if img_size:
            transformed_image = cv2.resize(transformed_image, img_size)

        return transformed_image
    else:
        if img_size:
            transformed_image = cv2.resize(image, img_size)
        return transformed_image



class TestImgAugDataset(ImageDataset):

    def preprocess(self, batch_x):
        batch_x_aug = [augment(cv2.imread(img_path), img_size=self.img_size) for img_path in batch_x]
        batch_x_aug += [cv2.resize(cv2.imread(img_path), self.img_size) for img_path in batch_x]
        return batch_x_aug

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

def age_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

weights="weights/age_pre_tf1.hdf5"##"age_net.hdf5"#"weights/age_resnet50.hdf5"#"weights/age_pre_tf1.hdf5"#"weights/age_resnet50.hdf5"
model = load_model(weights, custom_objects={"age_mae":age_mae, "age_loss":age_loss})

def visualize_data(dataset):
    for b_x,b_y in dataset:
        print("batch size: ", b_x.shape)
        x0, y = b_x[0], b_y[0]
        x1 = b_x[1]
        img_copy = np.copy(x0)

        pred = model.predict(np.array([x0]))
        print("Pred: ", np.argmax(pred))
        print("age: ", np.argmax(y))
        image = cv2.resize(x0,(200,200))
        cv2.imshow("img", image)
        #cv2.imshow("img_tf", transformed_image)
        cv2.waitKey(0)

        pred = model.predict(np.array([x1]))
        print("Pred: ", np.argmax(pred))
        print("age: ", np.argmax(y))
        image = cv2.resize(x1,(200,200))
        cv2.imshow("img", image)
        #cv2.imshow("img_tf", transformed_image)
        cv2.waitKey(0)
        """
        for x,y in zip(b_x, b_y):
            print("Shape: ", x.shape)
            # Augment an image
            pred = model.predict(np.array([x]))
            print("Pred: ", np.argmax(pred))
            print("age: ", np.argmax(y))
            image = cv2.resize(x,(500,500))
            cv2.imshow("img", image)
            #cv2.imshow("img_tf", transformed_image)
            cv2.waitKey(0)
        """

        #Smile face
        ##########################
        """
        try:
            img = img_copy
            lf = LandmarkFace.estimate(img)

            a_per_face = Pipeline([Smile(scale=0.3)])
            a_all = Multiple(a_per_face)

            new_lf, _ = a_all.perform(lf)
            cv2.imshow("img chubby", new_lf[0].img)
            pred = model.predict(np.array([new_lf[0].img]))
            print("Pred: ", np.argmax(pred))
            print("age: ", np.argmax(y))
            #cv2.imshow("img_tf", transformed_image)
            cv2.waitKey(0)
        except:
            pass
        """
        ##########################

if __name__=="__main__":
    
    dataset = TestImgAugDataset(DatasetIterator(csv_file="data/train.csv", num_classes=82, batch_size=1), base_path="../training_caip_contest/")
    
    #dataset.load_from_csv("data/train.csv")
    visualize_data(dataset)