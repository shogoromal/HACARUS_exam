from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
import datetime
import pytz
from sklearn.metrics import accuracy_score, f1_score

from base_class import Base, DataInputBody, DataRequestBody, ModelRequestBody, TrainRequestBody
from typing import List

from sql_manipulate import *
from utils import *
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
def DataUploader(passenger_datas: List[DataInputBody]):
    for pass_d in passenger_datas:
        if pass_d.Survived is not None:
            transaction_result, _ = add_passenger_data(pass_d)
    return {"transaction_result":transaction_result}

@app.post("/get_data")
def GetData(data_request:DataRequestBody):
    result_list = get_psg_cls_list(data_request.start_index, data_request.end_index)
    return result_list #sqlalchemyのBaseクラス、ただしリクエストの結果として返されるのはjsonを示す文字列のバイナリデータ

@app.post("/get_model")
def GetModel(data_request:ModelRequestBody):
    result_list = get_model_cls_list(data_request.version_id_1, data_request.version_id_2)
    return result_list #sqlalchemyのBaseクラス、ただしリクエストの結果として返されるのはjsonを示す文字列のバイナリデータ

@app.post("/train_new_model")
def TrainNewModel(data_request:TrainRequestBody):
    transaction_result, _, _ = make_new_model(data_request.start_index, data_request.end_index, data_request.my_model_name)
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
    if training_used_data_df_1 is None:
        return {"result":"指定した\" version_id_1\"のトレーニングデータがありません"}
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
    #推論のデータなどをプロットする
    plot_inference_result_graphs(target_psg_cls_list, inference_result_list, model_id_list, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare'], 'inference_result')
    
    return {"result":"結果の出力に成功しました。"}

@app.get("/reset_db")
def ResetDB():
    result3 = reset_db_rlt()
    result4 = reset_db_hcl()
    result1 = reset_db_psg()
    result2 = reset_db_mdp()
    result5 = reset_db_set()
    set_default_setting()
    return {"datas":result1, "models":result2, "data_model_relation":result3, "health_check_log":result4, "setting":result5}

@app.post("/setting")
def ChangeSetting(setting_request: SettingRequestBody):
    request = setting_request.dict()
    result = get_setting()
    result = result.toDict()
    if all((request[key] == result[key]) or (request[key] == None) for key in request.keys()):
        return {'result':'settingの更新はありません', 'setting':result}
    for key in request.keys():
        if request[key] == None:
            request[key] = result[key]
    res = change_setting(SettingRequestBody(**request))
    return {'result':res, 'setting':request}

@app.post("/pipeline")
def Pipeline(pipeline_request: List[DataInputBody]):
    
    #pipelineの設定を取得
    setting = get_setting()
    #使用するモデルを取得
    model, _ = get_trained_model(setting.best_model_id)
    
    #モデルがない場合の例外処理
    if model == None:
        return {'result':'モデルがありませんでした。データを登録して最初のモデルを登録してください。'}
    
    #あるモデルに対しての最初のデータかどうかで振り分けるために使うbool関数を定義
    pipeline_first_data = True
    #最後に表示する結果のリストを定義
    health_check_result = []
    prediction_result = []
    
    print(len(pipeline_request))

    for data in pipeline_request:
        
        #データに欠損がある場合の例外処理
        if data.input_check == True:
            data_dict = data.toDict()
            data_dict['result'] = 'データ欠損'
            print('データ欠損')
        
        #Survivedの情報を持つ場合は
        elif data.Survived != None:
            
            #データベースにデータを追加する
            _, psg_data = add_passenger_data(data)
            #入力データに対して、推論した結果を足す
            data_dict = data.toDict()
            data_dict['result'] = 1 if sigmoid(calculate_prediction_score(model, psg_data)) > 0.5 else 0
            
            #データ範囲を指定する変数を定義する
            start_index = psg_data.data_id - setting.num_for_result_check + 1
            end_index = psg_data.data_id
        
            #最初のデータの時のみ、num_for_result_num 個さかのぼってdata-modelの組み合わせの推論結果を登録する
            tmp_range = range(end_index, end_index + 1)
            if pipeline_first_data == True:
                tmp_range = range(start_index, end_index + 1)
                pipeline_first_data = False#次のデータからはこの処理は不要
                
            #直近の入力に対しての推論結果をデータベースに登録
            for check_id in tmp_range:
                check_psg_data = get_psg_cls_list(check_id, check_id + 1)
                prediction_score = calculate_prediction_score(model, check_psg_data[0])
                #第４引数がFalse->トレーニング時の推論ではないことを示す
                _, record = add_data_model_relation(prediction_score, setting.best_model_id, check_psg_data[0], False)
            
            #num_for_result_num 個さかのぼってdata-modelの組み合わせを取得する
            print('通常入力の際のデータ取得')
            print(start_index, end_index)
            tmp_accuracy_score, tmp_f1_score = get_model_evaluation_score(setting.best_model_id, start_index, end_index +1)
            
            #モデルを更新するかどうかの判断を、settingの値と計算値の比較から検証する
            NG_decision = (setting.check_type == "accuracy" and tmp_accuracy_score < setting.threshold_percentage)\
                or (setting.check_type == "f1" and tmp_f1_score < setting.threshold_percentage)
                            
            #データベースへhealth_checkの結果を追加する
            result, record = add_health_check_log(setting.best_model_id, psg_data.data_id, setting.setting_id, tmp_accuracy_score, tmp_f1_score, NG_decision)
            #最終的にresponseとして返すリストに追加する
            health_check_record_dict = record.toDict()
            
            #以下の変数について、前の変数が残らないようにする
            new_f1_score = None
            new_accuracy_score = None
            
            if NG_decision == True:
                
                #データ範囲を指定する変数を定義する
                model_train_start_index = psg_data.data_id - setting.num_for_re_training + 1
                model_train_end_index = psg_data.data_id
                
                #設定したデータ範囲から新しいモデルを作る
                transaction_result, model, model_version_id = make_new_model(model_train_start_index, model_train_end_index + 1, 'pipeline')
                
                #直近の入力に対しての推論結果をデータベースに登録
                for check_id in range(start_index, end_index + 1):
                    check_psg_data = get_psg_cls_list(check_id, check_id + 1)
                    prediction_score = calculate_prediction_score(model, check_psg_data[0])
                    #第４引数がFalse->トレーニング時の推論ではないことを示す
                    _, record = add_data_model_relation(prediction_score, model_version_id, check_psg_data[0], False)
                
                #評価スコアを計算する
                print('モデル更新時のデータ取得')
                new_accuracy_score, new_f1_score = get_model_evaluation_score(model_version_id, start_index, end_index + 1)
                
                if (setting.check_type == "accuracy" and tmp_accuracy_score <= new_accuracy_score)\
                    or (setting.check_type == "f1" and tmp_f1_score <= new_f1_score):
                
                    #settingのモデルを更新する
                    res = change_setting(SettingRequestBody(
                        best_model_id = model_version_id,
                        num_for_result_check = setting.num_for_result_check,
                        check_type = setting.check_type,
                        threshold_percentage = setting.threshold_percentage,
                        num_for_re_training = setting.num_for_re_training
                        ))
                    setting = get_setting()
                
                
            # returnする辞書型に項目を追加
            health_check_record_dict['new_accuracy_score'] = new_accuracy_score
            health_check_record_dict['new_f1_score'] = new_f1_score
            
            # 最終的に返すlistに追加
            health_check_result.append(health_check_record_dict)

        #Survivedの結果を持たないデータの場合
        else:
            psg_data = sql_PassengerData(      
                survived = None,
                upload_date = None,
                pclass = data.Plass,
                sex = data.Sex,
                age = data.Age,
                fare = data.Fare
                )
            data['result'] = 1 if sigmoid(calculate_prediction_score(model, psg_data)) > 0.5 else 0
        
        #データに推論の結果を追加する
        prediction_result.append(data_dict)
        
    return {'prediction_result':prediction_result, 'health_check_result':health_check_result}
