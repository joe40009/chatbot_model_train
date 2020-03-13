import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import models
import pandas as pd
import jieba
import logging
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional,LSTM,Dense,Embedding,Dropout,Activation,Softmax
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pymysql
import numpy
from opencc import OpenCC
import jieba.posseg as jp
import numpy
import argparse
import yaml
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

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
        list(csv_data['quest_nm_adj'])[i],
        list(csv_data['class_nm'])[i]])

print('-'*10 + 'head 5 data' + '-'*10)
for kw in kw_results[0:5]:
    print(kw)

class_list_orgn = []
for raw in kw_results:
    rsl = raw[1].split(',')
    for rs in rsl:
        class_list_orgn.append(rs)
unique, counts = numpy.unique(class_list_orgn, return_counts=True)
print('-'*10 + 'NER count' + '-'*10)
print(dict(zip(unique, counts)))

i = 0
label_dic = {}
for label_raw in unique:
#     print(label_raw[0])
    label_dic[label_raw] = i
    i += 1
class_list = []
for classes in class_list_orgn:
    class_list.append(label_dic[classes])
print('-'*10 + 'class2eng' + '-'*10)
print(label_dic) #把CLASS名稱轉成數字

data_list = []
for raw in kw_results:
    data_list.append(raw[0])
print('-'*10 + 'head 5 train data' + '-'*10)    
for dl in data_list[0:5]: #訓練句子資料
    print(dl)

cc= OpenCC('s2twp')
words_file = config['stopwords'] #文本檔案位置
stopwords_set = set()                                           #创建set集合
with open(words_file, 'r', encoding = 'utf-8') as f:            #打开文件
    for line in f.readlines():                                  #一行一行读取
        word = line.strip()                                     #去回车
        if len(word) > 0:                                       #有文本，则添加到words_set中
            stopwords_set.add(cc.convert(word))

jieba.load_userdict(config['jieba_dict']) #讀取自建辭庫

# 建立 W2V model
def train_word2vec(sentences,save_path):
    sentences_seg = []
    for i in sentences:
        sen_proc = []
        try:
            for word in jieba.cut(i):
                sen_proc.append(word)
            sentences_seg.append(sen_proc)
        except:
             continue                
    print('-'*10 + "start w2v train" + '-'*10) 
    model = Word2Vec(sentences_seg,
                size=100,  # 词向量维度
                min_count=5,  # 词频阈值
                window=5)  # 窗口大小    
    model.save(save_path)
    print('-'*10 + "w2v train complete" + '-'*10)
    return model, sentences_seg 

# train W2V
model, sentences_seg =  train_word2vec(data_list, config['w2v_save_path'])   

# W2V編號與權重
def generate_id2wec(word2vec_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model[word] for word in w2id.keys()}  # 词语的词向量
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2id,embedding_weights
# 句子轉成W2V編號
def text_to_array(w2index, senlist):  # 文本转为索引数字模式
    sentences_array = []
    for sen in senlist:
        new_sen = [ w2index.get(word,0) for word in sen]   # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)
# 準備訓練資料
def prepare_data(w2id,sentences,labels,max_len=200):
    X_train, X_val, y_train, y_val = train_test_split(sentences,labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), np_utils.to_categorical(y_train) ,np.array(X_val), np_utils.to_categorical(y_val)


# 產生W2V編號與權重
print('-'*10 + "generate_id2wec" + '-'*10) 
w2id,embedding_weights = generate_id2wec(model)

print('-'*10 + "prepare_data" + '-'*10) 
x_train,y_trian, x_val , y_val = prepare_data(w2id,sentences_seg,class_list,200)

# 建立 LSTM model
class Sentiment:
    def __init__(self,w2id,embedding_weights,Embedding_dim,maxlen,labels_category, save_model_path='./LSTM_model/LSTM_model/sentiment.h5'):
        self.Embedding_dim = Embedding_dim
        self.embedding_weights = embedding_weights
        self.vocab = w2id
        self.labels_category = labels_category
        self.maxlen = maxlen
        self.model = self.build_model()
        self.smp = save_model_path
        
    def build_model(self):
        model = Sequential()
        #input dim(140,100)
        model.add(Embedding(output_dim = self.Embedding_dim,
                           input_dim=len(self.vocab)+1,
                           weights=[self.embedding_weights],
                           input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50),merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.labels_category))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                     optimizer='adam', 
                     metrics=['accuracy'])
        model.summary()
        return model
    
    def train(self,X_train, y_train,X_test, y_test,n_epoch=5 ):
        self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch,
                      validation_data=(X_test, y_test))
        self.model.save(self.smp)   
        
    def predict(self,model_path,new_sen):
        model = self.model
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id =[ self.vocab.get(word,0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        return np.argmax(res)

# 訓練 LSTM
# Sentiment(w2id,embedding_weights, 词向量维度, 句子長度, class數目, model儲存位置')
senti = Sentiment(w2id, embedding_weights, 100 ,200 ,config['class_num'] , config['LSTM_save_path'])
# senti.train(x_train,y_trian, x_val ,y_val, Epoch次數)
print('-'*10 + "start LSTM train" + '-'*10) 
senti.train(x_train,y_trian, x_val ,y_val,config['epoch'])
print('-'*10 + "LSTM train complete" + '-'*10) 


if config['SMOTE']:

    # SMOTE資料增強
    # SMOTE
    print('-'*10 + "start SMOTE LSTM train" + '-'*10) 
    oversampler=SMOTE(random_state=45)
    x_resampled, y_resampled = oversampler.fit_sample(x_train, y_trian)
    senti = Sentiment(w2id,embedding_weights,100,200,config['class_num'], config['LSTM_save_path'][:-3] + '_SMOTE.h5')
    senti.train(x_resampled,y_resampled, x_val ,y_val,config['epoch'])
    print('-'*10 + "train complete" + '-'*10) 
    
    print('-'*10 + "start SMOTE kind=borderline-1 LSTM train" + '-'*10) 
    # SMOTE kind=borderline-1
    oversampler=BorderlineSMOTE(random_state=45, kind='borderline-1')
    x_resampled, y_resampled = oversampler.fit_sample(x_train, y_trian)
    senti = Sentiment(w2id,embedding_weights,100,200,config['class_num'], config['LSTM_save_path'][:-3] + '_SMOTEk1.h5')
    senti.train(x_resampled,y_resampled, x_val ,y_val,config['epoch'])
    print('-'*10 + "train complete" + '-'*10) 
    
    print('-'*10 + "start SMOTE kind=borderline-2 LSTM train" + '-'*10) 
    # SMOTE kind=borderline-2
    oversampler=BorderlineSMOTE(random_state=45, kind='borderline-2')
    x_resampled, y_resampled = oversampler.fit_sample(x_train, y_trian)
    senti = Sentiment(w2id,embedding_weights,100,200,config['class_num'], config['LSTM_save_path'][:-3] + '_SMOTEk2.h5')
    senti.train(x_resampled,y_resampled, x_val ,y_val,config['epoch'])
    print('-'*10 + "train complete" + '-'*10) 
    
    print('-'*10 + "start SMOTEENN LSTM train" + '-'*10) 
    # SMOTEENN
    oversampler=SMOTEENN(random_state=45)
    x_resampled, y_resampled = oversampler.fit_sample(x_train, y_trian)
    senti = Sentiment(w2id,embedding_weights,100,200,config['class_num'], config['LSTM_save_path'][:-3] + '_SMOTEENN.h5')
    senti.train(x_resampled,y_resampled, x_val ,y_val,config['epoch'])
    print('-'*10 + "train complete" + '-'*10) 
    
    print('-'*10 + "start SMOTETomek LSTM train" + '-'*10) 
    # SMOTETomek
    oversampler=SMOTETomek(random_state=45)
    x_resampled, y_resampled = oversampler.fit_sample(x_train, y_trian)
    senti = Sentiment(w2id,embedding_weights,100,200,config['class_num'], config['LSTM_save_path'][:-3] + '_SMOTETomek.h5')
    senti.train(x_resampled,y_resampled, x_val ,y_val,config['epoch'])
    print('-'*10 + "train complete" + '-'*10) 
    
    # ADASYN
    try:
        oversampler=ADASYN(random_state=45)
        x_resampled, y_resampled = oversampler.fit_sample(x_train, y_trian)
        print('-'*10 + "start ADASYN LSTM train" + '-'*10) 
        senti = Sentiment(w2id,embedding_weights,100,200,config['class_num'], config['LSTM_save_path'][:-3] + '_ADASYN.h5')
        senti.train(x_resampled,y_resampled, x_val ,y_val,config['epoch'])
        print('-'*10 + "train complete" + '-'*10) 
    except ValueError:
        print('ADASYN: No samples will be generated with the provided ratio settings.')
    
        
    print('-'*10 + "start SVMSMOTE LSTM train" + '-'*10) 
    # SVMSMOTE
    oversampler=SVMSMOTE(random_state=45)
    x_resampled, y_resampled = oversampler.fit_sample(x_train, y_trian)
    senti = Sentiment(w2id,embedding_weights,100,200,config['class_num'], config['LSTM_save_path'][:-3] + '_SVMSMOTE.h5')
    senti.train(x_resampled,y_resampled, x_val ,y_val,config['epoch'])
    print('-'*10 + "train complete" + '-'*10)



























