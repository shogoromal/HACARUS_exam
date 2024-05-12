from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from sklearn.linear_model import LogisticRegression


from dotenv import load_dotenv
import os
import datetime
import pytz

from base_model import PassengerData, DataIndex
from typing import List

from sql_manipulate import add_passenger_data, get_passenger_data, add_model_parameter, add_data_model_relation
from utils import sql_psg_class_to_dict, split_data, calculate_accuracy

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
def GetData(data_index:DataIndex):
    result_list = []
    results = get_passenger_data(data_index.start_index, data_index.end_index)
    for result in results:
        result_list.append(result)
    return result_list #sqlalchemyのBaseクラス、ただしリクエストの結果として返されるのはjsonを示す文字列のバイナリデータ

@app.post("/train_new_model")
def TrainNewModel(data_index:DataIndex):
    extracted_passenger_datas = GetData(data_index)
    df = sql_psg_class_to_dict(extracted_passenger_datas)
    X_train, X_test, y_train, y_test = split_data(df)
    #モデルをフィッティングする
    # Train the model and make predictions
    model = LogisticRegression()
    model.fit(X_train, y_train)
    transaction_result, model_version_id = add_model_parameter(model)
    
    """
    #各データに対してのpredict_scoreを登録していく
    for psg_data in extracted_passenger_datas:
        add_data_model_relation(model, model_version_id, psg_data, True)
    """
        
    return {"transaction_result":transaction_result, "accuracy":calculate_accuracy(model,X_test,y_test)}