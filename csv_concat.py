import os

import pandas as pd
import os
def FilesPath(path):
    filePaths = [] 
    for root,dirs,files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root,file))
    return filePaths

def train_TS(Total_file):
    data = pd.DataFrame()
    for file_path in (Total_file[0:76]):
        df = pd.read_csv(file_path)
        data = pd.concat([data,df.head(2)])
    data.to_csv('TS_train.csv')

def test_TS(Total_file):
    data = pd.DataFrame()
    for file_path in (Total_file[0:76]):
        df = pd.read_csv(file_path)
        data = pd.concat([data, df.head(3).tail(1)])
    data.to_csv('TS_test.csv')




if __name__ == '__main__':
    Total_file = FilesPath('C:\\Users\\zhao\\PycharmProjects\\gait\\gait_TS')
    # train_TS(Total_file)
    test_TS(Total_file)

