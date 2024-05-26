import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import attrgetter
from sklearn.metrics import accuracy_score, f1_score

from base_class import GraphSetting, sql_Relation, sql_PassengerData
from typing import List

from sql_manipulate import \
    get_model_parameter
from utils import \
    dataframe_to_dict, sql_psg_class_to_df, sigmoid
    
gs = GraphSetting()

def plot_psg_data(fig, psg_dict_lists:List[dict], model_id_list:List[str], graph_keys:List[str]):
    # 共通のカラムごとにヒストグラムor円グラフをプロット
    y_position = 2
    x_position = 1
    for key in graph_keys:
        if key in gs.hist_graph:
            ax = fig.add_subplot(len(graph_keys)+1, 1, y_position)
            n_bin = gs.bins.get(key) if gs.bins.get(key) is not None else 20
            color_index = 0
            for psg_dict_list, name in zip(psg_dict_lists, model_id_list):
                # データをプロット
                temp_list = [i[key] for i in psg_dict_list]
                print(key, len(temp_list))
                ax.hist(temp_list, label='model-'+str(name), alpha=.5, bins=n_bin, color=gs.colors[color_index], range= gs.range.get(key))
                color_index += 1
            plt.title(f'Histogram of {key}')
            plt.xlabel(key)
            plt.ylabel('Frequency')
            plt.legend()
            #plt.subplots_adjust()
        else:
            x_position = 0       
            for psg_dict_list, name in zip(psg_dict_lists, model_id_list):
                if len(model_id_list) == 2:
                    ax = fig.add_subplot(len(graph_keys)+1, 2, 2 * y_position + x_position - 1)
                elif len(model_id_list) == 1:
                    ax = fig.add_subplot(len(graph_keys)+1, 1, y_position)
                temp_list = [i[key] for i in psg_dict_list]
                list_component = set(temp_list)
                count_list = []
                for lc in list_component:
                    count_list.append(temp_list.count(lc))
                ax.pie(count_list, labels=list_component)
                x_position += 1
                plt.title(f'Piechart of {key} - model {name}')
                plt.legend()
        y_position += 1
        
def logit_scatter_plot(target_psg_cls_list:List[sql_PassengerData],global_p_list:List[int], model_id_list:List[int]):
    num = 4
    fig = plt.figure(figsize=(12,num*8))
    x_age, x_sex, x_pclass, x_fare = [[],[]], [[],[]], [[],[]], [[],[]]
    div_p_list = [[[],[]], [[],[]]]
    temp_p_list = []
    if len(global_p_list) == 2:
        for p_0, p_1 in zip(global_p_list[0], global_p_list[1]):
            temp_p_list.append([p_0, p_1])
    for tpc, p_list in zip(target_psg_cls_list,temp_p_list):
        x_age[tpc.survived].append(tpc.age)
        x_sex[tpc.survived].append(tpc.sex)
        x_pclass[tpc.survived].append(tpc.pclass)
        x_fare[tpc.survived].append(tpc.fare)
        div_p_list[0][tpc.survived].append(p_list[0])
        if len(global_p_list) == 2:
            div_p_list[1][tpc.survived].append(p_list[1])
    itr_list = [x_age, x_sex, x_pclass, x_fare]
    itr_name = ['age', 'sex', 'pclass', 'fare']
    x_position = 1
    for div_p in div_p_list:
        if div_p[0] == []:
            break    
        y_position = 1
        for itr, name in zip(itr_list,itr_name):
            if len(global_p_list) == 2:
                ax = fig.add_subplot(num*2, 2, 2 * y_position + x_position - 2)
            if len(global_p_list) == 1:
                ax = fig.add_subplot(num, 1, y_position)
            ax.scatter(itr[0], div_p[0], color='red', marker='x')
            ax.scatter(itr[1], div_p[1], color='blue', marker='x')
            ax.set_ylim(0,1)
            plt.title(f'Scatter of {name} - model {model_id_list[x_position-1]}')
            y_position += 1
        x_position += 1
        
    plt.savefig('outputs/'+ 'logit_scatter_plot' +'.jpg')
    
    return None
        
def plot_model_check_graphs(psg_dict_lists:List[dict], model_id_list:List[int], graph_keys:List[str], output_name:str):  
    fig = plt.figure(figsize=(10,len(graph_keys)*8))
    
    table_columns = ['','model-' + str(model_id_list[0]),'nothing']
    table = [['Age-coef','',''],
             ['Pclass-coef','',''],
             ['Sex-coef','',''],
             ['Fare-coef','',''],
             ['model_name','',''],
             ['train_data_num','','']]
    if len(model_id_list) == 2:
        table_columns[2] = 'model-' + str(model_id_list[1])
    #テーブルでデータを表示する
    column_count = 1
    for model_ver_id in model_id_list:
        model_cls = get_model_parameter(model_ver_id)
        table[0][column_count] = model_cls.age_coef
        table[1][column_count] = model_cls.pclass_coef
        table[2][column_count] = model_cls.sex_coef
        table[3][column_count] = model_cls.fare_coef
        table[4][column_count] = model_cls.my_model_name
        table[5][column_count] = len(psg_dict_lists[column_count-1])
        column_count += 1
    ax = fig.add_subplot(len(graph_keys)+1, 1, 1)
    ax.axis('off')
    table = ax.table(cellText=table,colLabels=table_columns, cellLoc='center',loc='center',colColours=['#909099', '#909099', '#909099'])
    table.set_fontsize(14)
    table.scale(0.8, 3)
    
    plot_psg_data(fig, psg_dict_lists, [str(i) for i in model_id_list], graph_keys)

    plt.savefig('outputs/'+output_name+'.jpg')

def plot_inference_result_graphs(target_psg_cls_list:List[sql_PassengerData], inference_result_list:List[sql_Relation], model_id_list:List[int], graph_keys:List[str], output_name:str):
    fig = plt.figure(figsize=(10,len(graph_keys)*8))
    
    ###tableを作る
    table_columns = ['','model-' + str(model_id_list[0]),'nothing']
    if len(model_id_list) == 2:
        table_columns[2] = 'model-' + str(model_id_list[1])
    table = [['Accuracy','',''],
             ['F1-score','','']]
    
    ###inference_result から予想クラスを算出してaccuracyを算出する
    #data_id順に並び替える
    target_psg_cls_list = sorted(target_psg_cls_list,key=attrgetter('data_id'))
    ground_truth_list = []
    #print(len(target_psg_cls_list))
    for tpc in target_psg_cls_list:
        ground_truth_list.append(tpc.survived)
        #print(tpc.data_id)
    
    #データをソートする
    for i in range(len(inference_result_list)):
        inference_result_list[i] = sorted(inference_result_list[i],key=attrgetter('data_id'))
    
    column_count = 1
    global_p_list = []
    for ir_list in inference_result_list:
        #print(len(ir_list))
        prediction_list = []
        p_list = []
        for ir in ir_list:
            p_list.append(sigmoid(ir.prediction_score))
            prediction_list.append(1 if sigmoid(ir.prediction_score)>0.5 else 0)
            #print(ir.data_id)

        table[0][column_count] = str(round(accuracy_score(ground_truth_list, prediction_list),2))
        table[1][column_count] = str(round(f1_score(ground_truth_list, prediction_list),2))
        column_count += 1
        global_p_list.append(p_list)
        
    ax = fig.add_subplot(len(graph_keys)+1, 1, 1)
    ax.axis('off')
    table = ax.table(cellText=table,colLabels=table_columns, cellLoc='center',loc='center',colColours=['#909099', '#909099', '#909099'])
    table.set_fontsize(14)
    table.scale(0.8, 3)
    
    target_psg_dict_list = dataframe_to_dict(sql_psg_class_to_df(target_psg_cls_list))
    target_psg_dict_list = [target_psg_dict_list]
    plot_psg_data(fig, target_psg_dict_list, ['target_data'], graph_keys)

    plt.savefig('outputs/'+output_name+'.jpg')
    
    logit_scatter_plot(target_psg_cls_list,global_p_list, model_id_list)
    
    return None
        