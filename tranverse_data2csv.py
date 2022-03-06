import pandas as pd
import os
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def trarnverse_label2num(data_file):
    num = len(data_file)
    return dict(zip(data_file,range(num)))

# 1 给数据集打上标签并创建字典表示 保存csv文件到data_csv
def tranverse_images():
    closed_eyes = []
    open_eyes = []
    labels = pd.DataFrame()
    closed_images = glob.glob('./Fatigue Detection/closed_eye/*.png')
    for image in closed_images:
        closed_eyes.append(image)
    open_images = glob.glob('./Fatigue Detection/open_eye/*.png')
    for image in open_images:
        open_eyes.append(image)
    all_features = np.hstack((closed_eyes,open_eyes))
    all_y = np.hstack((np.zeros(len(closed_eyes)),np.ones(len(open_eyes))))
    for url,label in zip(all_features,all_y):
        labels_data = pd.DataFrame({"data_url":[url],"labels":[label]})
        labels = pd.concat((labels,labels_data))
    labels = shuffle(labels)
    labels.head()
    labels.to_csv('Eys_Dataset_Csv.csv')

# 2 划分数据集 保存为train_data_csv 和 test_data_csv
# params : ratio 划分率  default = 0.8
def split_into_train_test(ratio=0.6):
    train_labels = pd.DataFrame()
    test_labels = pd.DataFrame()
    # 读取数据集表格 取消第一行作为表头 因为第一行表头会被算进去 所以遍历1:后的所有数据
    dataset = pd.read_csv('Eys_Dataset_Csv.csv')
    train_data,test_data = train_test_split(dataset,test_size=(1-ratio),random_state=11)
    for url,label in zip(train_data.data_url,train_data.labels):
        train_ = pd.DataFrame({"data_url":[url],"labels":[label]})
        train_labels = pd.concat((train_labels,train_))
    for url in test_data.data_url:
        test_ = pd.DataFrame({"data_url":[url]})
        test_labels = pd.concat((test_labels,test_))
    train_labels.head()
    train_labels.to_csv('Eys_Train_Dataset.csv')
    test_labels.head()
    test_labels.to_csv('Eys_Test_Dataset.csv')

if __name__ == "__main__":
    split_into_train_test()