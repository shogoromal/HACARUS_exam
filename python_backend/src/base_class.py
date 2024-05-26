from pydantic import BaseModel, validator
from typing import Union

from sqlalchemy import  Column, Integer, Float, Text, DateTime, Boolean 
from sqlalchemy.ext.declarative import declarative_base

# セッションを作成する
Base = declarative_base()

class sql_PassengerData(Base):
    __tablename__ = 'datas'
    data_id = Column(Integer, primary_key=True, autoincrement=True)
    upload_date = Column(DateTime)
    survived = Column(Integer, nullable=True)
    pclass = Column(Integer, nullable=True)
    sex = Column(Integer, nullable=True)
    age = Column(Float, nullable=True)
    fare = Column(Float, nullable=True)
    def toDict(self):
        return {
            'data_id': self.data_id,
            'upload_date': self.upload_date,
            'Survived': self.survived,
            'Pclass': self.pclass,
            'Sex': self.sex,
            'Age': self.age,
            'Fare': self.fare            
        }
class sql_ModelParameter(Base):
    __tablename__ = 'models'
    model_version_id = Column(Integer, primary_key=True, autoincrement=True)
    training_date = Column(DateTime)
    my_model_name = Column(Text)
    pclass_coef = Column(Float, nullable=False)
    sex_coef = Column(Float, nullable=False)
    age_coef = Column(Float, nullable=False)
    fare_coef = Column(Float, nullable=False)
    training_iteration = Column(Integer, nullable=False)
    def toDict(self):
        return{
            'mode]_version_id': self.model_version_id,
            'my_model_name': self.my_model_name,
            'pclass_coef': self.pclass_coef,
            'sex_coef': self.sex_coef,
            'age_coef': self.age_coef,
            'fare_coef': self.age_coef
        }
    
class sql_Relation(Base):
    __tablename__ = 'data_model_relations'
    experiment_id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, nullable=False)
    data_id = Column(Integer, nullable=False)
    include_training = Column(Boolean, nullable=False)
    prediction_score = Column(Float, nullable=False)
    
class sql_Setting(Base):
    __tablename__ = 'setting'
    setting_id = Column(Integer, primary_key=True)
    best_model_id = Column(Integer, nullable=False)
    num_for_result_check = Column(Integer, nullable=False)
    check_type = Column(Text,nullable=False)
    threshold_percentage = Column(Integer, nullable=False)
    num_for_re_training = Column(Integer, nullable=False)
    def toDict(self):
        return{
            'setting_id': self.setting_id,
            'best_model_id':self.best_model_id,
            'num_for_result_check':self.num_for_result_check,
            'check_type':self.check_type,
            'threshold_percentage':self.threshold_percentage,
            'num_for_re_training':self.num_for_re_training
        }
        
        
class sql_HealthCheckLog(Base):
    __tablename__ = 'health_check_log'
    health_check_id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, nullable=False)
    data_id = Column(Integer, nullable=False)
    setting_id = Column(Integer, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    NG_decision = Column(Boolean, nullable=False)
    def toDict(self):
        return {
            'health_check_id':self.health_check_id,
            'model_version_id':self.model_version_id,
            'data_id':self.data_id,
            'setting_id':self.setting_id,
            'accuracy_score':self.accuracy_score,
            'f1_score':self.f1_score,
            'NG_decision':self.NG_decision          
        }

class DataInputBody(BaseModel):
    Survived: Union[int, None]  = None
    Pclass: int
    Sex: int
    Age: float
    Fare: float
    def input_check(self):
        if self.Pclass == None or self.Sex == None or self.Age == None or self.Fare == None:
            return True
        return False
    def toDict(self):
        return{
            'Survived':self.Survived,
            'Pclass':self.Pclass,
            'Sex':self.Sex,
            'Age':self.Age,
            'Fare':self.Fare
        }

class DataRequestBody(BaseModel):
    start_index:int
    end_index:int

class TrainRequestBody(BaseModel):
    start_index:int
    end_index:int
    my_model_name:str = 'no_name'

class ModelRequestBody(BaseModel):
    version_id_1:int
    version_id_2:Union[int,None] = None
    start_index:int = None
    end_index:int = None

class SettingRequestBody(BaseModel):
    #setting_id:int
    best_model_id: Union[int,None] = None
    num_for_result_check: Union[int,None] = None
    check_type: Union[str,None] = None
    threshold_percentage: Union[float,None] = None
    num_for_re_training: Union[int,None] = None
    @validator("check_type")
    def age_size(cls, v):
        if v == "f1" or v == "accuracy" or v == None:
            return v
        else:
            raise ValueError("check_type には accuracy か f1 を入力してください。")
    @validator("threshold_percentage")
    def age_size(cls, v):
        if v == None:
            return v
        if 0 <= v <= 1:
            return v
        else:
            raise ValueError("threshold_ percentage は 0~1 の間で指定してください。")
        
class GraphSetting():
    def __init__(self):
        self.colors = ['blue','red','green','yellow']
        self.bins = {'Fare':25 ,'Age':10}
        self.range = {'Fare':(0,250), 'Age':(0,100)}
        self.hist_graph = ['Age', 'Fare']