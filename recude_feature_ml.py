from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time as ti

from sklearn.svm import SVC

Stand_X = StandardScaler()


def knn():
    model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
    threshold = 0.7
    return model,threshold

def rf():
    model=RandomForestClassifier(n_estimators=200, random_state=42)
    threshold = 0.5
    return model, threshold

def avova():
    known_data = pd.read_csv('new_gait_dataset/avova.csv')
    unknown_data = pd.read_csv('new_gait_dataset/avova_unknow.csv')
    X_known = known_data.iloc[:, :10]
    y_known = known_data.iloc[:, 10]

    # model, threshold = knn()
    model, threshold = rf()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    unknown_splits = np.array_split(unknown_data, 10)
    scores = []
    pre = []
    rec = []
    f1score = []
    for i, (train_index, test_index) in enumerate(kf.split(X_known)):
        # 训练集只包含已知类数据
        X_train, y_train = X_known.iloc[train_index], y_known.iloc[train_index]
        X_test_known, y_test_known = X_known.iloc[test_index], y_known.iloc[test_index]

        X_test = pd.concat([X_test_known, unknown_splits[i].iloc[:, :10]], ignore_index=True)
        y_test = pd.concat([y_test_known, unknown_splits[i].iloc[:, 10]], ignore_index=True)

        # 训练模型
        model.fit(X_train, y_train)
        # knn_threshold = 0.85
        # rf_threshold=0.55

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        max_prob = np.max(y_prob, axis=1)
        unknown_samples = np.max(y_prob, axis=1)
        y_pred[max_prob < threshold] = 51

        cm = confusion_matrix(y_test, y_pred)

        diam = np.trace(cm)
        sum = cm.sum()

        TN = cm[-1, -1]
        TP = diam - TN
        FP = cm[-1, :].sum() - TN
        FN = cm[:, -1].sum() - TN
        # FN=sum-TN-TP-FP

        accuracy = diam / sum
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        scores.append(accuracy)
        pre.append(precision)
        rec.append(recall)
        f1score.append(f1)
def cross_validation_fold(train_index, test_index, X_known, y_known, unknown_split,x):

    X_train, y_train = X_known.iloc[train_index], y_known.iloc[train_index]
    X_test_known, y_test_known = X_known.iloc[test_index], y_known.iloc[test_index]


    X_test = pd.concat([X_test_known, unknown_split.iloc[:, :10]], ignore_index=True)
    y_test = pd.concat([y_test_known, unknown_split.iloc[:, 10]], ignore_index=True)


    X_train = Stand_X.fit_transform(X_train)
    X_test = Stand_X.transform(X_test)


    model = SVC(C=21, kernel='rbf', decision_function_shape='ovr', probability=True)

    model.fit(X_train, y_train)

    threshold = 0.6
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    max_prob = np.max(y_prob, axis=1)
    unknown_samples = np.max(y_prob, axis=1)
    y_pred[max_prob < threshold] = 51



    cm = confusion_matrix(y_test, y_pred)

    diam = np.trace(cm)
    sum = cm.sum()

    TN = cm[-1, -1]
    TP = diam - TN

    FP = cm[-1, :].sum() - TN
    FN = cm[:, -1].sum() - TN
    # FN=sum-TN-TP-FP

    accuracy = diam / sum
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    y=[accuracy, precision, recall, f1]
    x.append(y)
    return x



def parallel_cross_validation(X_known, y_known, unknown_splits, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True,random_state=42)
    x=[]
    with Pool() as pool:
        y = pool.starmap(
            cross_validation_fold,
            [(train_index, test_index, X_known, y_known, unknown_splits[i],x)
             for i, (train_index, test_index) in enumerate(kf.split(X_known))]
        )
    return y


if __name__=='__main__':


    # avova()


    multisvm()
