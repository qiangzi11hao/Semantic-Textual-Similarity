# encoding: utf-8
# module unicodedata
# from F:\anaconda\Anaconda3\envs\TF\DLLs\unicodedata.pyd
# by generator 1.145
import nltk
from gensim.models import word2vec
def all_word(read_txt,save_path):
    '''
    整理所有数据
    :param read_txt: 读取所有数据
    :param save_path: 保存数据
    '''
    with open(read_txt,encoding='utf-8') as f:
        for line in f.readlines():
            str1=line.split('\t')[1]
            str2=line.split('\t')[2]
            if '\n' in str2:
                str2=str2.split('\n')[0]
            with open(save_path,'a',encoding='utf-8') as fin:
                fin.write(str1+'\n'+str2+'\n')

def build_model(read_txt,save_model):
    '''
    模型建立
    :param read_txt: 文件读取
    :param save_model: 模型保存
    '''
    sentence=word2vec.Text8Corpus(read_txt)
    model=word2vec.Word2Vec(sentence,size=100,min_count=1,window=2)
    model.save(save_model)

def similar(word,model_path):
    '''
    相似度检测
    '''
    model=word2vec.Word2Vec.load(model_path)
    print(model.most_similar(word))

def count(read_txt,glove_path):
    word_dic={}
    vector_dic=[]
    with open(glove_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')
            word_dic[line[0]]={}
            vector=[]
            for i in range(1,len(line)):
                vector.append(line[i])
            word_dic[line[0]]=vector
    dic=[]
    count=[]
    with open(read_txt,encoding='utf-8') as f:
        for line in f.readlines():
            line=nltk.word_tokenize(line.split('\n')[0])
            for word in line:
                if word not in dic:
                    dic.append(word)
                    if word in word_dic:
                        print(word_dic)
                    # with open('new_vec',encoding='utf-8') as f:
                    #     f.write(word)
                    # count.append(1)

    for i in range(len(count)):
        if len(dic[i])==1:
            print(dic[i])
        # print(dic[i],count[i])
    # print(len(count))


##各种地址，路径
train_txt='./data/train.txt'
test_txt='./data/test.txt'
all_text='all_word.txt'
glove_path='./data/dictionary/glove.840B.3000d.txt'
all_word(train_txt,all_text)
all_word(test_txt,all_text)

save_model_path='w2v_dic'
build_model(all_text,save_model_path)
similar('play',save_model_path)

