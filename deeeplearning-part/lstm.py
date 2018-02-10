
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Embedding,LSTM,Dense,Flatten
from keras.layers import BatchNormalization,Input,concatenate
vocal_size=2090
embedding_size=300
max_len=12
epoch=20
batch_size=16


def read_train_file(train_path,tokenizer):
    '''
    读取训练文件
    :param train_path: 训练文件地址
    :param tokenizer: 字典序
    :return: 处理好的训练文件
    '''
    train_txt1=[]
    train_txt2=[]
    train_label=[]
    with open(train_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split('\t')
            train_txt1.append((line[1]))
            train_txt2.append((line[2]))
            train_label.append(line[len(line)-1])

    train_txt1=tokenizer.texts_to_sequences(train_txt1)
    train_txt2=tokenizer.texts_to_sequences(train_txt2)
    train_txt1=pad_sequences(train_txt1,maxlen=max_len)
    train_txt2=pad_sequences(train_txt2,maxlen=max_len)
    train_label=np.array(train_label)
    print(train_txt2.shape)
    return train_txt1,train_txt2,train_label

def read_test_file(test_path,tokenizer):
    '''
    :param test_path: 测试文件地址
    :param tokenizer: 字典
    :return: 处理好的测试数据
    '''
    test_x1=[]
    test_x2=[]
    with open(test_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split('\t')
            test_x1.append(line[1])
            test_x2.append(line[2])
    test_x1=tokenizer.texts_to_sequences(test_x1)
    test_x2=tokenizer.texts_to_sequences(test_x2)
    test_x1 = pad_sequences(test_x1, maxlen=max_len)
    test_x2 = pad_sequences(test_x2, maxlen=max_len)
    return test_x1,test_x2


def create_embedding_matrix(vec_path):
    '''
    创建字典向量矩阵及字典
    :param vec_path: 向量路径
    :return: 向量矩阵及字典
    '''
    word_vec={}
    word_dic=[]
    # word_dic.append('#')
    with open(vec_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split(' ')
            word_vec[line[0]]={}
            # print(line[0],line[1])
            vec=[]
            for i in range(1,embedding_size+1):
                vec.append(float(line[i]))
            word_vec[line[0]]=vec
            word_dic.append(line[0])
    embedding_matrix=np.zeros((vocal_size+1,embedding_size))

    tokenizer=Tokenizer(num_words=vocal_size,lower=False)
    tokenizer.fit_on_texts(word_dic)
    word_index=tokenizer.word_index
    print(len(tokenizer.word_index))
    for word,i in word_index.items():
        embedding_matrix[i]=word_vec[word]
    # print(embedding_matrix)
    return embedding_matrix,tokenizer

def build_model(embedding_matrxi):
    '''
        模型构建，可以用LSTM或CNN，这里演示代码句子1使用CNN，句子2使用LSTM，分别x1，x2表示
        :param embedding_matrxi:词向量矩阵
        :return: 模型
    '''
    from keras.layers import Conv1D, MaxPool1D
    sequence_input1 = Input(shape=(max_len,), dtype='int32')
    sequence_input2 = Input(shape=(max_len,), dtype='int32')
    embedding_layer = Embedding(vocal_size + 1, embedding_size,
                                weights=[embedding_matrxi],
                                input_length=max_len, trainable=False)
    embedded_sequence = embedding_layer(sequence_input1)
    x1 = Conv1D(128, 5, activation='relu')(embedded_sequence)
    x1 = Flatten()(x1)

    embedded_sequence = embedding_layer(sequence_input2)
    x2 = Conv1D(128, 5, activation='relu')(embedded_sequence)
    x2 = Flatten()(x2)
    x2 = LSTM(50)(x2)
    x2 = Dense(64)(x2)
    merge = concatenate([x1, x2])
    merge = BatchNormalization()(merge)
    merge = Dense(100)(merge)
    merge = BatchNormalization()(merge)
    preds = Dense(1, activation='relu')(merge)

    model = Model(inputs=[sequence_input1, sequence_input2], outputs=preds)
    model.compile(loss='mse', optimizer='nadam', metrics=['mse'])
    return model

def result_file(save_path,test_path,preds):
    '''
    结果文件保存，强制规则大于5就为5，控制范围
    :param save_path:保存地址
    :param test_path: 测试文件
    :param preds: 得分预测
    '''
    id_list=[]
    with open(test_path,encoding='utf-8') as f:
        for line in f.readlines():
            id_list.append(line.split('\t')[0])
    with open(save_path,'w',encoding='utf-8') as f:
        for i in range(len(id_list)):
            if preds[i][0]>5:
                preds[i][0]=5.0

            f.write(str(id_list[i])+','+str(preds[i][0])+'\n')


model_path='model_save'
train_path='./data/train.txt'
test_path='./data/test.txt'
vec_path='./new_vec2.txt'
save_path='submission_sample'
create_embedding_matrix('new_vec2.txt')
matrix,tokenizer=create_embedding_matrix(vec_path=vec_path)
train_x1,train_x2,train_label=read_train_file(train_path,tokenizer)

val_x1=train_x1[1250:]
val_x2=train_x2[1250:]
val_label=train_label[1250:]

model=build_model(matrix)

# 模型画图
from keras.utils import plot_model
plot_model(model, to_file='model.png')
# 模型数据喂入
model.fit([train_x1,train_x2],train_label,epochs=epoch,batch_size=batch_size,
          validation_data=([val_x1,val_x1],val_label),shuffle=True)
