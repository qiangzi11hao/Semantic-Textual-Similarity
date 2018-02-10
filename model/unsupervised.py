

import nltk
from nltk.corpus import wordnet
word_dic={}

def load_file(embedding_path,file_path,result_save):
    '''
    读入文件
    :param embedding_path: 词向量文件
    :param file_path: 文件路径
    :param result_save: 结果保存文件
    '''
    count=0
    with open(embedding_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split(' ')
            word_dic[line[0]]={}
            vector=[]
            for i in range(1,len(line)):
                vector.append(float(line[i]))
            word_dic[line[0]]=vector
    with open(result_save, 'w', encoding='utf-8') as fin:
        with open(file_path,encoding='utf-8') as f:
            for line in f.readlines():
                line=line.split('\t')
                str1=line[1]
                str2=line[2]
                if '\n' in str2:
                    str2=str2.split('\n')[0]
                score=unspupervised(str1,str2)
                fin.write(line[0]+' '+str(score)+'\n')
                count+=1
                # if count==10:break




def much_function(word,anther_str):
    '''
    match 函数，参见论文
    :param word: 当前词
    :param anther_str: 对应的另外的句子
    :return: 最大得分
    '''
    import numpy as np
    if word in anther_str:
        return 1
    else:
        sim = 0
        for i in wordnet.synsets(word):
            syn_word=i.lemmas()[0].name()
            if syn_word in anther_str:
                if word in word_dic:
                    A=np.array(word_dic[word])
                else:
                    A=np.ones((300)) # 300维词向量


                if syn_word in word_dic:
                    B=np.array(word_dic[syn_word])
                else:
                    B=np.ones((300))
                Lx = np.sqrt(A.dot(A))
                Ly = np.sqrt(B.dot(B))
                cos = A.dot(B) / (Lx * Ly)
                if cos>sim:
                    sim=cos
        return sim



def unspupervised(str1,str2):
    '''
    非监督算法执行
    :param str1:句子1
    :param str2: 句子2
    :return: 相似度得分
    '''
    str1=nltk.word_tokenize(str1)
    str2=nltk.word_tokenize(str2)
    word_score = 0
    for word in str1:
        word_score+=much_function(word,str2)

    for word in str2:
        word_score+=much_function(word,str1)
    return word_score/float(len(str2)+len(str1))



train_path='./data/train.txt'
embedding_path='./data/new_vec.txt'
train_result='train_score.txt'
test_path='./data/test.txt'
test_result='test_score.txt'
load_file(embedding_path,test_path,test_result)