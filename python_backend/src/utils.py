from typing import List
from base_model import PassengerData

from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

def dataframe_to_dict(df:pd.DataFrame):
    try:
        result_list = []
        df = df[['Survived','Pclass','Sex','Age','Fare']]
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

def sql_psg_class_to_dict(passenger_data:List[PassengerData]):
    # sqlalchemyのBaseモデルのリストを辞書のリストに変換
    dict_list = [item.toDict() for item in passenger_data]
    # 辞書のリストをDataFrameに変換
    df = pd.DataFrame(dict_list)
    print(df)
    return df

def split_data(df:pd.DataFrame):
    df = df.dropna()
    # Split the data into features and target
    X = df[['Survived','Pclass','Sex','Age','Fare']].drop('Survived', axis=1)
    y = df['Survived']
    # Split the data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def calculate_accuracy(model,X_test,y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)