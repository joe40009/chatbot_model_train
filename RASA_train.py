import pymysql
import csv
import pandas as pd
import numpy
import json
from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
import argparse
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

with open(args.config, 'r') as stream:
    config = yaml.load(stream)


if config['mode'] == 'sql':
    sql = config['sql'] #SQL指令拉資料
    db = pymysql.connect(host=config['sql_ip'], port=config['sql_port'], user=config['sql_user'], passwd=config['sql_passwd'], db=config['sql_db'] )
    cursor = db.cursor()
    cursor.execute(sql)
    kw_results = cursor.fetchall() #從MYSQL拉出資料

if config['mode'] == 'csv':
    csv_data = pd.read_csv(config['csv_data'], encoding=config['csv_encoding']) #從CSV拉資料
    kw_results = [] #整理CSV資料
    for i in range(len(csv_data)):
        kw_results.append([
        list(csv_data['class_nm'])[i],
        list(csv_data['quest_nm_adj'])[i],
        list(csv_data['quest_kw'])[i],
        list(csv_data['quest_ner'])[i]]) 

print('-'*10 + 'head 5 data' + '-'*10)
for kw in kw_results[0:5]:
    print(kw)        
        
class_list_orgn = []
for raw in kw_results:
    rsl = raw[3].split(',')
    for rs in rsl:
        class_list_orgn.append(rs)
unique, counts = numpy.unique(class_list_orgn, return_counts=True)
print('-'*10 + 'NER count' + '-'*10)
print(dict(zip(unique, counts)))

class_list_orgn = []
for raw in kw_results:
    rsl = raw[0].split(',')
    for rs in rsl:
        class_list_orgn.append(rs)
unique, counts = numpy.unique(class_list_orgn, return_counts=True)
print('-'*10 + 'class count' + '-'*10)
print(dict(zip(unique, counts)))

class2eng = config['class2eng']
dil = []

dataall = {"rasa_nlu_data": {"common_examples":[]}}

for testtext in kw_results:

    data = {}
    data["entities"] = []
    if len(class2eng) > 1:
        data["intent"] = class2eng[testtext[0]]
    else:
        data["intent"] = testtext[3].split(',')[0]
    try:

        if len(testtext[3].split(',')) < 2:
            nerl = testtext[3]
            for kwl in testtext[2].split(','):

                entities = {"start": testtext[1].index(kwl), 
                             "end": testtext[1].index(kwl) + len(kwl), 
                             "value": kwl, 
                             "entity":testtext[3]}

                data["entities"].append(entities)
        else:
#                     kwl = testtext[2].split(',')
            nernm = 0
            nerl = testtext[3].split(',')
            for kwl in testtext[2].split(','):


                entities = {"start": testtext[1].index(kwl), 
                             "end": testtext[1].index(kwl) + len(kwl), 
                             "value": kwl, 
                             "entity":nerl[nernm]}

                data["entities"].append(entities)                   
                nernm += 1

    except (ValueError, IndexError):    
        continue

    data["text"] = testtext[1]

    dataall["rasa_nlu_data"]["common_examples"].append(data)
    
print('-'*10 + 'head 5 rasa data' + '-'*10)
for rd in dataall["rasa_nlu_data"]["common_examples"][0:5]:
    print(rd)

print('-'*10 + 'save rasa data json' + '-'*10)
json_path = os.path.join(config['model_path'], config['model_name']) + '.json'
with open(json_path, 'w') as f: #儲存成JSON格式
    json.dump(dataall, f)

print('-'*10 + 'rasa train' + '-'*10)
#RASA訓練
from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer

training_data = load_data(json_path) #RASA訓練資料位置(JSON)
trainer = Trainer(config.load("./nlu_config.yml")) #RASA訓練設定
trainer.train(training_data)
model_directory = trainer.persist(config['model_path'], fixed_model_name=config['model_name']) #model儲存位置     

print('-'*10 + 'rasa train complete' + '-'*10)
        
        
        
        
        
        
        
        
        
        
