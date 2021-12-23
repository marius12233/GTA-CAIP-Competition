import pandas as pd
import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence, to_categorical
import os
import sys
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from utils.augment_dataset import augment
import random

class Data:

    def __init__(self, X, y):
        assert len(X)==len(y)
        self.x = X
        self.y = y
    
    def __len__(self):
        return len(self.x)

class Dataset:
    def __init__(self, data: Data=None, shuffle=True, csv_file=None, num_classes=None, to_categorical=True):
        self.shuffle = shuffle
        self.data=data
        self.num_classes = num_classes
        self.to_categorical = to_categorical
        self.df = None
        if csv_file and self.data is None:
            self._load_from_csv(csv_file)
        if num_classes:
            self.num_classes = num_classes
        
        
    def _load_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        if not self.num_classes:
            self.num_classes = len(df.label.unique())+1
        if self.shuffle:
            df = shuffle(df)
        self.df=df
        X = df["image"].values
        y = df["label"].values
        if self.to_categorical:
            y = to_categorical(y, num_classes = self.get_num_classes())
        self.data = Data(X,y)
    

    def balance_dataset(self, max_occ_class=3000):
        random.seed(27)
        labels = self.data.y
        filtered_idxs = []
        classes_counts = self.get_classes_counts()
        for class_ in classes_counts.keys():
            count = classes_counts[class_]
            idxs = np.where(labels==class_)[0]
            if count>max_occ_class:
                #choise max_occ_class indexes from class class_
                sampled_idxs = random.sample(idxs.tolist(),max_occ_class)
                filtered_idxs+=sampled_idxs
            else:
                filtered_idxs+=idxs.tolist()
        #obtained the filter indexes, apply on X,y
        filtered_idxs = shuffle(filtered_idxs)
        self.data.x = self.data.x[filtered_idxs]
        self.data.y = self.data.y[filtered_idxs]

    def labels_to_categorical(self):
        y = to_categorical(self.data.y, num_classes = self.get_num_classes())
        self.data.y = y

    def get_num_classes(self):
        return self.num_classes
    
    def get_dataframe(self):
        return self.df
    
    def __len__(self):
        return len(self.data.x)
    
    def get_classes_counts(self):
        if self.to_categorical:
            raise Exception("categorical classes, use a not categorical transformation dataset")
        classes_count = {}
        labels = self.data.y
        for i in range(1,82):
            classes_count[i] = sum(labels==i)
        return classes_count

class DatasetIterator(Dataset):

    def __init__(self, data: Data=None, shuffle=True, csv_file=None, batch_size=32, num_classes=None, to_categorical=True):
        self.bs = batch_size
        self.current_index=0
        super().__init__(data, shuffle, csv_file, num_classes, to_categorical)
    
    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        return self.get_next_batch()
    
    def get_next_batch(self):
        batch_x = self.data.x[self.current_index:self.current_index+self.bs]
        batch_y = self.data.y[self.current_index:self.current_index+self.bs]
        self.current_index = (self.current_index + self.bs)%len(self.data)
        return batch_x, batch_y
    
    def get_batch_size(self):
        return self.bs
    
    def __len__(self):
        return len(self.data.x)//self.bs


#Decorator
class ImageDataset(DatasetIterator):

    def __init__(self, dataset, base_path=None, img_size=(224,224)):
        self.base_dataset = dataset
        self.base_path = base_path
        self.img_size = img_size
    
    def __iter__(self):
        iter(self.base_dataset)
        return self

    def __next__(self):
        return self.get_next_batch()

    def preprocess(self, batch_x):
        batch_x = [cv2.resize(cv2.imread(img_path), self.img_size) for img_path in batch_x]
        return batch_x    

    def get_next_batch(self):
        batch_x, batch_y = next(self.base_dataset)
        
        if self.base_path: #significa che l'elemento è un PATH
            batch_x = [i + j for i, j in zip([self.base_path]*len(batch_x), batch_x)]
        
        batch_x = self.preprocess(batch_x)
        return np.array(batch_x), np.array(batch_y)   

    def get_num_classes(self):
        return self.base_dataset.get_num_classes() 
    
    def get_batch_size(self):
        return self.base_dataset.get_batch_size()
    
    def __len__(self):
        return len(self.base_dataset)


class ImgAugDataset(ImageDataset):


    def preprocess(self, batch_x):

        batch_x = [augment(cv2.resize(cv2.imread(img_path), self.img_size), img_size=self.img_size) for img_path in batch_x]
        #batch_x = [cv2.resize(cv2.imread(img_path), self.img_size) for img_path in batch_x]
        return batch_x

from joblib import Parallel, delayed
import itertools as it

def chunks(data, size):
    #data = iter(data)
    for i in range(0, len(data), size):
        yield data[i]


class ImgAugMTDataset(ImgAugDataset):

    def __init__(self, dataset, base_path=None, img_size=(224,224)):
        self.base_dataset = dataset
        self.base_path = base_path
        self.img_size = img_size
        self.current_batch = super().get_next_batch()
    
    def get_next_batch(self):
        batch_x, batch_y = next(self.base_dataset)
        
        if self.base_path: #significa che l'elemento è un PATH
            batch_x = [i + j for i, j in zip([self.base_path]*len(batch_x), batch_x)]
        
        batch_x = self.preprocess(batch_x)
        current_batch = self.current_batch
        self.current_batch = np.array(batch_x), np.array(batch_y)

        return current_batch

    def preprocess(self, batch_x):
        with Parallel(n_jobs=len(batch_x)) as parallel:
            result = parallel(delayed(augment)(cv2.resize(cv2.imread(img_path), self.img_size), img_size=self.img_size) for img_path in chunks(batch_x, 1))
        #batch_x = [cv2.resize(cv2.imread(img_path), self.img_size) for img_path in batch_x]
        return result    

class EqualBatchIterator(DatasetIterator):
    pass

from threading import Lock

class KerasGenerator(Sequence):
    
    def __init__(self, dataset, batch_size=32, img_size=(224,224)):
        self.mutex = Lock()
        self.dataset=dataset
        self.batch_size = batch_size
        self.cur_index = 0
        self.base_path = "C:/Users/mario/training_caip_contest/"#"../training_caip_contest/"
        self.img_size=img_size
        self.on_epoch_end()
    
    def __getitem__(self, index):
        self.mutex.acquire()
        if self.cur_index >= len(self.dataset):
            #raise StopIteration
            self.cur_index = 0
        i = self.cur_index
        self.cur_index += self.batch_size
        self.mutex.release()
        data = self._load_batch(i)
        
        return tuple(data)

    def on_epoch_end(self):
        self.mutex.acquire()
        self.cur_index = 0
        self.mutex.release()
        #shuffle data
        data=self.dataset.data
        idxs = [i for i in range(0, len(data.x))]
        idxs = shuffle(idxs)
        self.dataset.data.x = data.x[idxs]
        self.dataset.data.y = data.y[idxs]
        print("Shuffled")


    def __len__(self):
        return len(self.dataset)//self.batch_size
    
    def preprocess(self, batch_x):
        return [cv2.resize(cv2.imread(img_path), self.img_size) for img_path in batch_x]
    
    def _load_batch(self, start_index):
        batch_x = self.dataset.data.x[start_index:start_index+self.batch_size]
        batch_y = self.dataset.data.y[start_index:start_index+self.batch_size]
        batch_x = [i + j for i, j in zip([self.base_path]*len(batch_x), batch_x)]
        batch_x = self.preprocess(batch_x)
        
        return np.array(batch_x)/255., np.array(batch_y)

class KerasAugGenerator(KerasGenerator):

    def preprocess(self, batch_x):
        batch_x = [augment(cv2.resize(cv2.imread(img_path), self.img_size), bgr=False, img_size=self.img_size) for img_path in batch_x]
        #batch_x = [cv2.resize(cv2.imread(img_path), self.img_size) for img_path in batch_x]
        return batch_x        


if __name__=="__main__":
    dataset = Dataset(csv_file="data/train.csv", num_classes=101, to_categorical=False)
    #dataset.load_from_csv("data/train.csv")
    print(type(dataset))
    print("# classes ", dataset.get_num_classes())
    #print(dataset.get_classes_counts())
    dataset.balance_dataset()
    print("After balancing")
    print(dataset.get_classes_counts())

    """
    for x,y in dataset:
        print(x[0].shape)
        break
    """

