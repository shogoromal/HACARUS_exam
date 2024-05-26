from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine, desc, func, and_, or_
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.ext.declarative import declarative_base

import os
from dotenv import load_dotenv
import datetime
import pytz
import numpy as np
import pandas as pd

from base_class import *
from typing import List

jst = pytz.timezone('Asia/Tokyo')#タイムゾーンを日本に設定
load_dotenv()  # .env ファイルを読み込む

user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
db_name = 'mydatabase'

# engineの設定
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{db_name}')

# セッションの作成
db_session = scoped_session(
  sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
  )
)


#新しいデータを追加する
def add_passenger_data(passenger_data:DataInputBody):
  try:
    new_record = sql_PassengerData(
      survived = passenger_data.Survived,
      upload_date = datetime.datetime.now(jst),
      pclass = passenger_data.Pclass,
      sex = passenger_data.Sex,
      age = passenger_data.Age,
      fare = passenger_data.Fare
      )
    db_session.add(new_record)
    db_session.commit()
    return "success", new_record
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false", None
  
#データベースからデータを取得する
def get_passenger_data(data_index_list:List[int]):
  try:
    
    if data_index_list == [-1]:
      return db_session.query(sql_PassengerData).all()
    
    query = db_session.query(sql_PassengerData).filter(
      or_(sql_PassengerData.data_id.in_(data_index_list))
      )
    return query.all()
  except Exception as e:
    print(e)
    db_session.rollback()
    return None
  
  
#学習したモデルを登録する
def add_model_parameter(model, model_name:str='no_name'):
  try:
    name_list = np.array(model.feature_names_in_)
    new_record = sql_ModelParameter(
      training_date = datetime.datetime.now(jst),
      my_model_name = model_name,
      pclass_coef = model.coef_[0][np.where(name_list=='Pclass')[0][0]],
      sex_coef = model.coef_[0][np.where(name_list=='Sex')[0][0]],
      age_coef = model.coef_[0][np.where(name_list=='Age')[0][0]],
      fare_coef = model.coef_[0][np.where(name_list=='Fare')[0][0]],
      training_iteration = int(model.n_iter_[0])
      )
    db_session.add(new_record)
    db_session.commit()
    return "success", new_record.model_version_id
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false", None
  
#モデルのパラメータを取得
def get_model_parameter(model_version_id:int=-1):
  try:
    if model_version_id == -1:
          query = db_session.query(sql_ModelParameter)
          return query.all()
    else:
      query = db_session.query(sql_ModelParameter).filter(
        sql_ModelParameter.model_version_id == model_version_id
        )
    return query.first()
  except Exception as e:
    print(e)
    db_session.rollback()
    return None
  
  
def add_data_model_relation(prediction_score:float, model_version_id:int, psg_data:sql_PassengerData, include_training=False):
  if include_training==False:
    query = db_session.query(sql_Relation).filter(
      sql_Relation.model_version_id == model_version_id,
      sql_Relation.data_id == psg_data.data_id
      ).first()
    if query != None:
      return "すでにmodel-dataの組み合わせで推論済み", None
  try:
    new_record = sql_Relation(
      model_version_id = model_version_id,
      data_id = psg_data.data_id,
      include_training = include_training,
      prediction_score = prediction_score,
      )
    db_session.add(new_record)
    db_session.commit()
    return "success", new_record
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false", None
  
#データを取得、include_training->Trueはトレーニング時の推論結果登録
def get_data_model_relation(model_version_id:int,include_training:bool=False, start_index:int=None, end_index:int=None):
  bool_filters = [sql_Relation.include_training.in_([True])]
  index_filters = []
  if include_training == False:
    bool_filters = [sql_Relation.include_training.in_([True, False])]
    data_index_list = [i for i in range(start_index, end_index)] #データの範囲を指定する必要があるのは、トレーニングデータを集める時以外
    index_filters = [sql_Relation.data_id.in_(data_index_list)]
    print('model_version_id', model_version_id)
    print('index_filters', data_index_list)
  try:
    query = db_session.query(sql_Relation).filter(
      sql_Relation.model_version_id == model_version_id,
      or_(*bool_filters),
      or_(*index_filters)
      )
    return query.all()
  except Exception as e:
    print(e)
    db_session.rollback()
    return None

#settingの変更をする
def change_setting(setting_request:SettingRequestBody):
  try:
    new_record = sql_Setting(
      best_model_id = setting_request.best_model_id,
      num_for_result_check = setting_request.num_for_result_check,
      check_type = setting_request.check_type,
      threshold_percentage = setting_request.threshold_percentage,
      num_for_re_training = setting_request.num_for_re_training
      )
    db_session.add(new_record)
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"
  
#データベースからデータを取得する
def get_setting():
  try:
    query = db_session.query(sql_Setting).order_by(desc('setting_id')).first()
    return query
  except Exception as e:
    print(e)
    db_session.rollback()
    return None
  
  
def add_health_check_log(model_version_id:int, data_id:int, setting_id:int, accuracy_score:float, f1_score:float, NG_decision:bool):
  try:
    new_record = sql_HealthCheckLog(
      model_version_id = model_version_id,
      data_id = data_id,
      setting_id = setting_id,
      accuracy_score = accuracy_score,
      f1_score = f1_score,
      NG_decision = NG_decision
      )
    db_session.add(new_record)
    db_session.commit()
    return "success", new_record
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false", None
#データを取得
def get_health_check_log():
  try:
    query = db_session.query(sql_HealthCheckLog)
    return query.all()
  except Exception as e:
    print(e)
    db_session.rollback()
    return None


#データベースの中身をすべて削除する
def reset_db_rlt():
  try:
    db_session.query(sql_Relation).delete()
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"
def reset_db_mdp():
  try:
    db_session.query(sql_ModelParameter).delete()
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"
def reset_db_psg():
  try:
    db_session.query(sql_PassengerData).delete()
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"
def reset_db_hcl():
  try:
    db_session.query(sql_HealthCheckLog).delete()
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"
def reset_db_set():
  try:
    db_session.query(sql_Setting).delete()
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"
def set_default_setting():
  try:
    new_record = sql_Setting(
      best_model_id = 1,
      num_for_result_check = 1,
      check_type = "accuracy",
      threshold_percentage = 0.5,
      num_for_re_training = 40
      )
    db_session.add(new_record)
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"