from pydantic import BaseModel
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
    age = Column(Integer, nullable=True)
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

class PassengerData(BaseModel):
    Survived: Union[int, None]  = None
    Pclass: Union[int, None]  = None
    Sex: Union[int, None]  = None
    Age: Union[int, None]  = None
    Fare: Union[float, None]  = None

class DataRequestBody(BaseModel):
    start_index:int = None
    end_index:int = None
    my_model_name:str = 'no_name'

class ModelRequestBody(BaseModel):
    version_id_1:int
    version_id_2:int = None
    start_index:int = None
    end_index:int = None

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
    
class sql_Relation(Base):
    __tablename__ = 'data_model_relations'
    experiment_id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, nullable=False)
    data_id = Column(Integer, nullable=False)
    include_training = Column(Boolean, nullable=False)
    prediction_score = Column(Float, nullable=False)