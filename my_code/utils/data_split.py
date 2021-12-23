import pandas as pd
import random
import os
import time

CSV_PATH="../training_caip_contest.csv"
TEST_PERCENTAGE = 0.1
#df_train, df_val = None, None

def split_random(group):
    n_test=int(round(group.count()*TEST_PERCENTAGE).values[0])
    test_group=group.sample(n=n_test, random_state=1)
    train_group = group[group.apply(lambda x: x.values.tolist() not in test_group.values.tolist(), axis=1)]
    train_csv = "train.csv"
    test_csv = "test.csv"

    if os.path.isfile(train_csv):
        train_group = pd.concat([train_group, pd.read_csv(train_csv, names=["image", "label"])])
    train_group.to_csv(train_csv)

    if os.path.isfile(test_csv):
        train_group = pd.concat([test_group, pd.read_csv(test_csv, names=["image", "label"])])
    test_group.to_csv(test_csv)

def split_random_group(group, df_train, df_test): 
    n_test=int(round(len(group)*TEST_PERCENTAGE))
    test_group=random.sample(group, k=n_test)
    train_group = [x for x in group if x not in test_group] 
    df_train+=train_group
    df_test+=test_group 


def split_data(csv_path=CSV_PATH):
    #read data
    start = time.time()
    df=pd.read_csv(csv_path, names=["image", "label"])
    df_group = df.groupby("label")
    #df_n_test = round(df_group.count()*test_percentage)

    #df_group.apply(split_random)
    df_train = []
    df_test = []

    groups = df_group.groups

    for k_group in groups.keys():
        group = groups[k_group].tolist()
        #print(group)
        split_random_group(group, df_train, df_test)

    df_train = df.iloc[df_train]
    df_test = df.iloc[df_test]

    df_train.to_csv("data/train.csv")
    df_test.to_csv("data/val.csv")
    

    end = time.time()
    print("Time: ", end-start)

    
if __name__=="__main__":
    split_data()
    #test_split()