from typing import List
from base_class import DataInputBody, DataRequestBody, ModelRequestBody, sql_PassengerData, sql_Setting

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sql_manipulate import *


################
###データ変換系###
################

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

def sql_psg_class_to_df(passenger_data:List[DataInputBody]):
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
    if test_size != 0:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    return X_train, y_train


################
###データ取得系###
################

#index番号からdataを取得
def get_psg_cls_list(start_index:int, end_index:int):
    if end_index == -1:
        data_index_list = [-1]
        results = get_passenger_data(data_index_list)
        return results
    data_index_list = [i for i in range(start_index, end_index)]
    results = get_passenger_data(data_index_list)
    return results

#モデルの情報を取得する
def get_model_cls_list(id_1:int, id_2:int=None):
    result = get_model_parameter(id_1)
    result_list = []
    result_list.append(result)
    if id_2 is not None:
        result = get_model_parameter(id_2)
        result_list.append(result)
    return result_list 

def get_training_used_datas(model_version_id:int):
    results = get_data_model_relation(model_version_id, True)#第二引数がTrueでトレーニング時に使ったデータを抽出する
    # resultsが空の場合はNoneを返す
    if not results:
        return None
    data_index_list = [re.data_id for re in results] 
    psg_datas = get_passenger_data(data_index_list)
    df = sql_psg_class_to_df(psg_datas)
    return df


################
###モデル構築系###
################

def make_trained_model(df:pd.DataFrame, test_size=0):
    if test_size == 0:
        X_train, y_train = validation_training_data(df,test_size) #test_size=0で指定したデータ全てをトレーニングに使う
    else:
        X_train, X_test, y_train, y_test = validation_training_data(df,test_size)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def get_trained_model(version_id:int):
    #モデルidからトレーニング時に使ったデータを取得、dfで返される
    training_used_data_df = get_training_used_datas(version_id)
    #取得できない場合はエラーを返す
    if training_used_data_df is None:
        return None, None
    #再度、トレーニングを実施してモデルを再現する
    model = make_trained_model(training_used_data_df)
    return model, training_used_data_df

def make_new_model(start_index:int, end_index:int, my_model_name:str):
        #トレーニングに使うデータ抽出、データフレームへの変換
    extracted_passenger_datas = get_psg_cls_list(start_index, end_index)
    if extracted_passenger_datas == []:
        return {"transaction_result":"指定したデータがありません。"}
    df = sql_psg_class_to_df(extracted_passenger_datas)
    model = make_trained_model(df)
    transaction_result, model_version_id = add_model_parameter(model,my_model_name)
    #各データに対してのpredict_scoreを登録していく
    #第４引数はTrueでトレーニング時に登録されたデータであることを示す
    for psg_data in extracted_passenger_datas:
        prediction_score = calculate_prediction_score(model, psg_data)
        add_data_model_relation(prediction_score, model_version_id, psg_data, True)
    return transaction_result, model, model_version_id


########################
###その他、計算ツールなど###
########################

def calculate_prediction_score(model,  psg_data:sql_PassengerData):
    #psg_dict = {key:[value] for key, value in psg_data.toDict().items()}
    #psg_df = pd.DataFrame(psg_dict)[['Pclass','Sex','Age','Fare']]
    psg_df = sql_psg_class_to_df([psg_data])
    psg_df = psg_df[['Pclass','Sex','Age','Fare']]
    prediction_score = model.decision_function(psg_df)[0]
    return prediction_score

def inference_to_new_data(version_id:int, start_index:int, end_index:int, datas_for_analysis:List[DataInputBody]):
    ###推論していないdataに対して推論を行う###
    #モデルとトレーニングに使ったデータを取得
    model, training_used_data_df = get_trained_model(version_id)
    #取得できない場合はエラーを返す
    if training_used_data_df is None:
        return None, None
    #「分析に使うデータ」に対してモデルを使った推論を行い、データベースに登録する
    for psg_data in datas_for_analysis:
        prediction_score = calculate_prediction_score(model, psg_data)
        add_data_model_relation(prediction_score, version_id, psg_data, False)#第４引数がFalse->トレーニング時の推論ではないことを示す
    ###推論結果を格納したデータベースから結果を取得する###
    inference_results = get_data_model_relation(version_id,False,start_index,end_index)
    return training_used_data_df, inference_results

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#指定したインデックスとmodelに対しての評価結果のリストを作る
def get_model_evaluation_score(model_id:int, start_index:int, end_index:int):
    
    #num_for_result_num 個さかのぼってdata-modelの組み合わせを取得する
    relation_cls_list = get_data_model_relation(model_id, False, start_index, end_index)
    
    #該当する入力データを取得する
    psg_cls_list = get_psg_cls_list(start_index, end_index)
    
    #予測値とラベルのリストを作る
    prediction_list = [1 if sigmoid(ps.prediction_score)>0.5 else 0 for ps in relation_cls_list]
    label_list = [label.survived for label in psg_cls_list]
    
    #accuracyとF1スコアを計算する
    tmp_accuracy_score = accuracy_score(label_list, prediction_list)
    tmp_f1_score = f1_score(label_list, prediction_list)
    
    return tmp_accuracy_score, tmp_f1_score