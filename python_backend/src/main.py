from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
import datetime
import pytz

from base_class import Base, PassengerData, DataRequestBody, ModelRequestBody
from typing import List

from sql_manipulate import \
    engine, add_passenger_data, add_model_parameter, add_data_model_relation, reset_db_rlt, reset_db_mdp, reset_db_psg
from utils import \
    sql_psg_class_to_df, make_trained_model, dataframe_to_dict, get_psg_cls_list, get_model_cls_list, inference_to_new_data
from visualize import\
    plot_model_check_graphs, plot_inference_result_graphs

jst = pytz.timezone('Asia/Tokyo')#タイムゾーンを日本に設定
load_dotenv() 

app = FastAPI()

#origin = os.getenv("FRONT_ENDPOINT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
@app.post("/upload_data")
def DataUploader(passenger_datas: List[PassengerData]):
    for pass_d in passenger_datas:
        transaction_result = add_passenger_data(pass_d)
    return {"transaction_result":transaction_result}

@app.post("/get_data")
def GetData(data_request:DataRequestBody):
    result_list = get_psg_cls_list(data_request.start_index, data_request.end_index)
    return result_list #sqlalchemyのBaseクラス、ただしリクエストの結果として返されるのはjsonを示す文字列のバイナリデータ

@app.post("/get_model")
def GetData(data_request:ModelRequestBody):
    result_list = get_model_cls_list(data_request.version_id_1, data_request.version_id_2)
    return result_list #sqlalchemyのBaseクラス、ただしリクエストの結果として返されるのはjsonを示す文字列のバイナリデータ

@app.post("/train_new_model")
def TrainNewModel(data_request:DataRequestBody):
    extracted_passenger_datas = get_psg_cls_list(data_request.start_index, data_request.end_index)
    df = sql_psg_class_to_df(extracted_passenger_datas)
    model = make_trained_model(df)
    transaction_result, model_version_id = add_model_parameter(model,data_request.my_model_name)
    
    #各データに対してのpredict_scoreを登録していく
    #第４引数はTrueでトレーニング時に登録されたデータであることを示す
    for psg_data in extracted_passenger_datas:
        add_data_model_relation(model, model_version_id, psg_data, True)
        
    return {"transaction_result":transaction_result}

@app.post("/evaluate_model")
def GetModelInfo(model_request:ModelRequestBody):
    
    ######2つのモデルの比較を行う######
    #histを作成する関数に渡すリストを定義する
    model_id_list = [model_request.version_id_1]
    tr_usd_psg_dict_lists = []
    #分析に使うデータを取得する
    target_psg_cls_list = get_psg_cls_list(model_request.start_index, model_request.end_index)
    #分析に使うデータに対して推論し、トレーニング時のデータも取得する
    training_used_data_df_1, inference_result_1 = inference_to_new_data(model_request.version_id_1,model_request.start_index,model_request.end_index,target_psg_cls_list)
    tr_usd_psg_dict_lists.append(dataframe_to_dict(training_used_data_df_1))
    #モデルidの２つ目が指定されている場合は、そちらに対しても上記の操作を行う
    inference_result_2 = None
    if model_request.version_id_2 is not None:
        training_used_data_df_2, inference_result_2 = inference_to_new_data(model_request.version_id_2,model_request.start_index,model_request.end_index,target_psg_cls_list)
        if training_used_data_df_2 is not None:
            model_id_list.append(model_request.version_id_2)
            tr_usd_psg_dict_lists.append(dataframe_to_dict(training_used_data_df_2))
    #modelを比較するグラフを出力
    plot_model_check_graphs(tr_usd_psg_dict_lists, model_id_list, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare'], 'compare_two_models_training_data')
    
    ######指定した範囲のdataに対しての推論の結果をプロットする######
    #推論の結果をリストにまとめる
    inference_result_list = [inference_result_1]
    if inference_result_2 is not None:
        inference_result_list.append(inference_result_2)
    
    plot_inference_result_graphs(target_psg_cls_list, inference_result_list, model_id_list, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare'], 'inference_result')
    
    return None

@app.get("/reset_db")
def ResetDB():
    result1 = reset_db_psg()
    result2 = reset_db_mdp()
    result3 = reset_db_rlt()
    return [result1, result2, result3]