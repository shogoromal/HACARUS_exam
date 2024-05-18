from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
import datetime
import pytz

from base_class import PassengerData, DataRequestBody, ModelRequestBody
from typing import List

from sql_manipulate import \
    add_passenger_data, add_model_parameter, add_data_model_relation
from utils import \
    sql_psg_class_to_df, make_trained_model, dataframe_to_dict, get_psg_cls_list, inference_to_new_data
from visualize import\
    plot_histograms

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

@app.post("/get_model_info")
def GetModelInfo(model_request:ModelRequestBody):
    #histを作成する関数に渡すリストを定義する
    model_id_list = [model_request.version_id_1]
    df_list = []
    #分析に使うデータを取得する
    datas_for_analysis = get_psg_cls_list(model_request.start_index, model_request.end_index)
    #右のデータに対して推論し、トレーニング時のデータも取得するin
    training_used_data_df_1, inference_result_1 = inference_to_new_data(model_request.version_id_1,model_request.start_index,model_request.end_index,datas_for_analysis )
    df_list.append(training_used_data_df_1)
    ###モデルidの２つ目が指定されている場合は、そちらに対しても上記の操作を行う
    if model_request.version_id_2 is not None:
        training_used_data_df_2, inference_result_2 = inference_to_new_data(model_request.version_id_2,model_request.start_index,model_request.end_index,datas_for_analysis )
        model_id_list.append(model_request.version_id_2)
        df_list.append(training_used_data_df_2)
    #histを作成する
    plot_histograms(df_list, model_id_list, 'compare_two_models_training_data')
    return None