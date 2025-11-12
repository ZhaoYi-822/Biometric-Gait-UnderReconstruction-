import pandas as pd
from sklearn.model_selection import train_test_split


def data_random_split():
    data=pd.read_csv('new_gait_dataset/unknow_data.csv')
    x_ts_data=data.iloc[:,:13]
    y_ts_label_data=data.iloc[:,13]

    x_train, x_test, y_train, y_test = train_test_split(x_ts_data, y_ts_label_data, train_size=0.2)
    x_train.to_csv('new_gait_dataset/original_vaild_gait_dataset.csv', index=False)
    print(x_train)


def data_split():
    df = pd.read_csv('new_dataset/original_gait_dataset.csv')
    sampled_df_60 = df.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.6)]).reset_index(drop=True)
    sampled_df_60.to_csv('new_dataset/train_gait_dataset.csv', index=False)
    sampled_df_40 = df.groupby('0').apply(lambda x: x.iloc[int(len(x) * 0.6):]).reset_index(drop=True)
    sampled_df_40_part1 = sampled_df_40.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.5)]).reset_index(drop=True)
    sampled_df_40_part2 = sampled_df_40.groupby('0').apply(lambda x: x.iloc[int(len(x) * 0.5):]).reset_index(drop=True)
    sampled_df_40_part1.to_csv('new_dataset/test_gait_dataset.csv', index=False)
    sampled_df_40_part2.to_csv('new_dataset/valid_gait_dataset.csv', index=False)
    print(sampled_df_40_part2)

def unknown_split():
    df = pd.read_csv('new_dataset/unknow_data.csv')
    sampled_df_p1 =df.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.2)]).reset_index(drop=True)
    sampled_df_p1.to_csv('new_dataset/unknow_vaild_data.csv', index=False)

    df=pd.read_csv('new_dataset/feature_unknown_class.csv')
    sampled_df_p2 = df.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.2)]).reset_index(drop=True)
    sampled_df_p2.to_csv('new_dataset/feature_unknow_vaild_data.csv', index=False)


if __name__ == '__main__':
   # data_split()
   # unknown_split()
   data_random_split()
