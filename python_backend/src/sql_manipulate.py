from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine, desc, func, and_
from sqlalchemy.orm.exc import NoResultFound

import os
from dotenv import load_dotenv
import datetime
import pytz
import numpy as np
import pandas as pd

from base_model import PassengerData, sql_PassengerData, sql_ModelParameter, sql_Relation

jst = pytz.timezone('Asia/Tokyo')#タイムゾーンを日本に設定
load_dotenv()  # .env ファイルを読み込む

user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
db_name = os.getenv('DATABASE')

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
def add_passenger_data(passenger_data:PassengerData):
  try:
    new_record = sql_PassengerData(
      survived = passenger_data.Survived,
      upload_date = datetime.datetime.now(jst),
      pclass = passenger_data.Pclass,
      sex = passenger_data.Sex,
      age = passenger_data.Age,
      fare = passenger_data.Fare,
      )
    db_session.add(new_record)
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"
  
#データベースからデータを取得する
def get_passenger_data(start_index:int,end_index:int):
  try:
    query = db_session.query(sql_PassengerData).filter(
      and_(sql_PassengerData.data_id > start_index-1, sql_PassengerData.data_id < end_index+1)
      )
    return query.all()
  except Exception as e:
    print(e)
    db_session.rollback()
    return None
  
#学習したモデルを登録する
def add_model_parameter(model):
  try:
    name_list = np.array(model.feature_names_in_)
    new_record = sql_ModelParameter(
      training_date = datetime.datetime.now(jst),
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
  
def add_data_model_relation(model, model_version_id:int, psg_data:sql_PassengerData, include_training=False):
  psg_dict = {key:[value] for key, value in psg_data.toDict().items()}
  psg_df = pd.DataFrame(psg_dict)[['Pclass','Sex','Age','Fare']]
  prediction_score = model.decision_function(psg_df)[0]
  try:
    new_record = sql_Relation(
      model_version_id = model_version_id,
      data_id = psg_data.data_id,
      include_training = include_training,
      prediction_score = prediction_score
      )
    db_session.add(new_record)
    db_session.commit()
    return "success"
  except Exception as e:
    print(e)
    db_session.rollback()
    return "false"