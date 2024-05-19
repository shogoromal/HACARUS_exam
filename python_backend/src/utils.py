from typing import List
from base_class import PassengerData, DataRequestBody, ModelRequestBody

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sql_manipulate import \
    get_data_model_relation, get_passenger_data, add_data_model_relation

def dataframe_to_dict(df:pd.DataFrame, mode:str='neutral'):
    try:
        result_list = []
        df = df[['Survived','Pclass','Sex','Age','Fare']]
        if mode=='sex_to_num':
            df.loc[:,'Sex'] = df.loc[:,'Sex'].map({'male': 0, 'female': 1})
        df = df.dropna()
        df.loc[:,'Age'] = df.loc[:,'Age'].astype(int)
        for i in range(len(df)):
            row = df.iloc[i]
            if row.isna().any():
                break  # NaNが見つかったらループを抜ける
            result_list.append(row.to_dict())
        return result_list
    except KeyError as e:
        print(e)
        return "正しいデータ形式ではありません。"
    except Exception as e:
        print(e)

def sql_psg_class_to_df(passenger_data:List[PassengerData]):
    # sqlalchemyのBaseモデルのリストを辞書のリストに変換
    dict_list = [item.toDict() for item in passenger_data]
    # 辞書のリストをDataFrameに変換
    df = pd.DataFrame(dict_list)
    return df

def validation_training_data(df:pd.DataFrame, test_size=0):
    df = df.dropna()
    # Split the data into features and target
    X_train = df[['Survived','Pclass','Sex','Age','Fare']].drop('Survived', axis=1)
    y_train = df['Survived']
    # Split the data into training set and test set
    if test_size is not 0:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    return X_train, y_train
"""
def calculate_accuracy(model,X_test,y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)
"""
def make_trained_model(df:pd.DataFrame, test_size=0):
    if test_size == 0:
        X_train, y_train = validation_training_data(df,test_size) #test_size=0で指定したデータ全てをトレーニングに使う
    else:
        X_train, X_test, y_train, y_test = validation_training_data(df,test_size)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

#index番号からdataを取得
def get_psg_cls_list(start_index:int, end_index:int):
    data_index_list = [i for i in range(start_index, end_index)]
    result_list = []
    results = get_passenger_data(data_index_list)
    for result in results:
        result_list.append(result)
    return result_list

def get_model_cls_list(id_1:int, id_2:int=None):
    results = get_passenger_data(id_1)
    result_list = []
    for result in results:
        result_list.append(result)
    if id_2 is not None:
        results = get_passenger_data(id_2)
        for result in results:
            result_list.append(result)
    return result_list 

def get_training_used_datas(model_version_id:int):
    results = get_data_model_relation(model_version_id, True)#第二引数がTrueでトレーニング時に使ったデータを抽出する
    data_index_list = [re.data_id for re in results] 
    psg_datas = get_passenger_data(data_index_list)
    df = sql_psg_class_to_df(psg_datas)
    return df

def inference_to_new_data(version_id:int, start_index:int, end_index:int, datas_for_analysis:List[PassengerData]):
    ###推論していないdataに対して推論を行う###
    #モデルidからトレーニング時に使ったデータを取得、dfで返される
    training_used_data_df = get_training_used_datas(version_id)
    #再度、トレーニングを実施してモデルを再現する
    model = make_trained_model(training_used_data_df)
    #「分析に使うデータ」に対してモデルを使った推論を行い、データベースに登録する
    for psg_data in datas_for_analysis:
        add_data_model_relation(model, version_id, psg_data, False)#第４引数がFalse->トレーニング時の推論ではないことを示す
    ###推論結果を格納したデータベースから結果を取得する###
    inference_results = get_data_model_relation(version_id,False,start_index,end_index)
    return training_used_data_df, inference_results

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))