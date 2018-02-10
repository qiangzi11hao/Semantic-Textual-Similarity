from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.corpus import brown, wordnet
from gensim import corpora, models, similarities
from Preprocessing import loadfile
import pickle
import nltk
import numpy as np
import re


"""
提取预训练的词向量，并定义常用的函数接口
"""
with open('glove.txt', 'r') as f:
    embeddings = {}
    for line in f.readlines():
        args = [i.strip('., ') for i in (line.strip("\n\t ")).split(' ')]
        embeddings[args[0]] = [float(i) for i in args[1:]]
wordnet_lemmatizer = WordNetLemmatizer()
tagger = nltk.tag.pos_tag
frequency_list = FreqDist(i.lower() for i in brown.words())
all_words_count = 0
for i in frequency_list:
    all_words_count += frequency_list[i]


"""
特征提取
"""
def vertorlize(content):
    vectorizer = CountVectorizer()
    X = vectorizer.fit(content)
    return X.toarray()


def bag_of_words(sen1, sen2):
    """
    :param sen1:
    :param sen2:
    :return: 两者的余弦相似度
    """
    return cosine_similarity([sen1], [sen2])[0][0]


def topic_id(all_sens):
    """
    使用LDA求所有句子的topic vector，每个句子都转化为一个n_topics维的向量
    :param all_sens:
    :return: len(all_sens)*6 的向量
    """
    lda = LatentDirichletAllocation(n_topics=6,
                                    learning_offset=50.,
                                    random_state=0)
    docres = lda.fit_transform(all_sens)
    return docres


def tf_idf(sen1, sen2):
    """
    :param sen1:
    :param sen2:
    :return: 求两者的tfidf向量
    """
    #     print("s1:",sen1,"s2:",sen2)
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform([sen1, sen2]).toarray()
    return tf_idf[0], tf_idf[1]


def lcs_dp(input_x, input_y):
    """
    :param input_x:
    :param input_y:
    :return: 求最大公共子字符串数
    """
    # input_y as column, input_x as row
    dp = [([0] * len(input_y)) for i in range(len(input_x))]
    maxlen = maxindex = 0
    for i in range(0, len(input_x)):
        for j in range(0, len(input_y)):
            if input_x[i] == input_y[j]:
                if i != 0 and j != 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if i == 0 or j == 0:
                    dp[i][j] = 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    maxindex = i + 1 - maxlen
                    # print('最长公共子串的长度是:%s' % maxlen)
                    # print('最长公共子串是:%s' % input_x[maxindex:maxindex + maxlen])
    return maxlen, input_x[maxindex:maxindex + maxlen]


def not_empty(s):
    """
    判断句子是否为空
    :param s:
    :return: True or False
    """
    return s and s.strip()


def tag_and_parser(str_sen1, str_sen2):
    """
    求两者的pos标注的最大公共子字符串,并分别除以两者的长度
    :param str_sen1:
    :param str_sen2:
    :return:
    """
    sen1 = list(filter(not_empty, str_sen1.split(" ")))
    sen2 = list(filter(not_empty, str_sen2.split(" ")))
    # print("sen1:",sen1)
    post_sen1 = nltk.pos_tag(sen1)
    post_sen2 = nltk.pos_tag(sen2)
    pos1, pos2 = "", ""
    # print(post_sen1,post_sen2)
    for word, pos in post_sen1:
        pos1 += pos + " "
    for word, pos in post_sen2:
        pos2 += pos + " "
    # print(pos1,pos2)
    maxlen, subseq = lcs_dp(pos1, pos2)
    return len(subseq.split(" ")) / len(str_sen1.split(' ')), len(subseq.split(" ")) / len(str_sen2.split(' '))



def get_ngram(word, n):
    """
    charater ngrams 特征提取
    :param word:
    :param n:
    :return:
    """
    ngrams = []
    word_len = len(word)
    for i in range(word_len - n + 1):
        ngrams.append(word[i: i + n])
    return ngrams


def get_lists_intersection(s1, s2):
    """
    求公共词
    :param s1:
    :param s2:
    :return:
    """
    s1_s2 = []
    for i in s1:
        if i in s2:
            s1_s2.append(i)
    return s1_s2


def overlap(sentence1_ngrams, sentence2_ngrams):
    """
    通过ngrams数学公式判断两个句子的相似度
    :param sentence1_ngrams:
    :param sentence2_ngrams:
    :return:
    """
    s1_len = len(sentence1_ngrams)
    s2_len = len(sentence2_ngrams)
    if s1_len == 0 and s2_len == 0:
        return 0
    s1_s2_len = max(1, len(get_lists_intersection(sentence2_ngrams, sentence1_ngrams)))
    return 2 / (s1_len / s1_s2_len + s2_len / s1_s2_len)


def get_ngram_feature(sentence1, sentence2, n):
    """
    指定n的大小,这里我们主要取的是n={1,2,3}
    :param sentence1:
    :param sentence2:
    :param n:
    :return:
    """
    sentence1_ngrams = []
    sentence2_ngrams = []

    for word in sentence1:
        sentence1_ngrams.extend(get_ngram(word, n))

    for word in sentence2:
        sentence2_ngrams.extend(get_ngram(word, n))

    return overlap(sentence1_ngrams, sentence2_ngrams)


def is_subset(s1, s2):
    for i in s1:
        if i not in s2:
            return False
    return True

#
# def get_numbers_feature(sentence1, sentence2):
#     s1_numbers = [float(i) for i in re.findall(r"[-+]?\d+\.?\d*", " ".join(sentence1))]
#     s2_numbers = [float(i) for i in re.findall(r"[-+]?\d+\.?\d*", " ".join(sentence2))]
#     s1_s2_numbers = []
#     for i in s1_numbers:
#         if i in s2_numbers:
#             s1_s2_numbers.append(i)
#
#     s1ands2 = max(len(s1_numbers) + len(s2_numbers), 1)
#     return [np.log(1 + s1ands2), 2 * len(s1_s2_numbers) / s1ands2,
#             is_subset(s1_numbers, s2_numbers) or is_subset(s2_numbers, s1_numbers)]


# def get_shallow_features(sentence):
#     counter = 0
#     for word in sentence:
#         if len(word) > 1 and (re.match("[A-Z].*]", word) or re.match("\.[A-Z]+]", word)):
#             counter += 1
#     return counter


def get_word_embedding(inf_content, word):
    """
    通过idf对word加权
    :param inf_content:
    :param word:
    :return:
    """
    if inf_content:
        return np.multiply(information_content(word), embeddings.get(word, np.zeros(300)))
    else:
        return embeddings.get(word, np.zeros(300))


def sum_embeddings(words, inf_content):
    """
    对sentence进行映射
    :param words:
    :param inf_content: 决定加权与否
    :return:
    """
    vec = get_word_embedding(inf_content, words[0])
    for word in words[1:]:
        vec = np.add(vec, get_word_embedding(inf_content, word))
    return np.array(vec)


def word_embeddings_feature(sentence1, sentence2):
    """
    :param sentence1:
    :param sentence2:
    :return: 不加权embedding 后的cosine similarity
    """
    return cosine_similarity(unpack(sum_embeddings(sentence1, False)),
                             unpack(sum_embeddings(sentence2, False)))[0][0]


def information_content(word):
    """
    :param word:
    :return: idf of word
    """
    return np.log(all_words_count / max(1, frequency_list[word]))


def unpack(param):
    return param.reshape(1, -1)


def weighted_word_embeddings_feature(sentence1, sentence2):
    """
    :param sentence1:
    :param sentence2:
    :return: 加权embedding 后的cosine similarity
    """
    return cosine_similarity(unpack(sum_embeddings(sentence1, True)),
                             unpack(sum_embeddings(sentence2, True)))[0][0]


def weighted_word_coverage(s1, s2):
    """
    :param s1:
    :param s2:
    :return: 计算交叉信息熵
    """
    s1_s2 = get_lists_intersection(s1, s2)
    return np.sum([information_content(i) for i in s1_s2]) / np.sum([information_content(i) for i in s2])


def harmonic_mean(s1, s2):
    """
    :param s1:
    :param s2:
    :return: s1和s2的调和平均
    """
    if s1 == 0 or s2 == 0:
        return 0
    return s1 * s2 / (s1 + s2)


def get_wordnet_pos(treebank_tag):
    """
    :param treebank_tag:
    :return: pos
    """
    if treebank_tag.startswith('A') or treebank_tag.startswith('JJ'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def get_synset(word):
    """
    :param word:
    :return: word的近义词
    """
    try:
        return wordnet.synset(word + "." + get_wordnet_pos(tagger([word])[0][1]) + ".01")
    except:
        return 0


def wordnet_score(word, s2):
    """
    :param word:
    :param s2:
    :return: word 和 sentence2 的相似程度
    """
    if word in s2:
        return 1
    else:
        similarities = []
        for w in s2:
            try:
                value = get_synset(word).path_similarity(get_synset(w))
                if value is None:
                    value = 0
                similarities.append(value)
            except AttributeError:
                similarities.append(0)
        return np.max(similarities)


def wordnet_overlap(s1, s2):
    """
    :param s1:
    :param s2:
    :return: s1和s2的相似程度
    """
    suma = 0
    for w in s1:
        suma += wordnet_score(w, s2)
    return suma / len(s2)


def feature_vector(a, b):
    fvec = []
    # Ngram overlap

    fvec.append(get_ngram_feature(a, b, 1))
    fvec.append(get_ngram_feature(a, b, 2))
    fvec.append(get_ngram_feature(a, b, 3))
    fvec.append(get_ngram_feature(a, b, 4))


    # WordNet-aug. overlap -
    fvec.append(harmonic_mean(wordnet_overlap(a, b), wordnet_overlap(b, a)))

    # Weighted word overlap -
    fvec.append(harmonic_mean(weighted_word_coverage(a, b),
                              weighted_word_coverage(b, a)))
    # sentence num_of_words differences -
    fvec.append(abs(len(a) - len(b)))

    # summed word embeddings - lagano
    fvec.append(word_embeddings_feature(a, b))
    fvec.append(weighted_word_embeddings_feature(a, b))

    # # # Shallow NERC - lagano
    # fvec.append(get_shallow_features(a))
    # fvec.append(get_shallow_features(b))
    #
    # # Numbers overlap - returns list of 3 features
    # fvec.extend(get_numbers_feature(a, b))
    # print(fvec)
    return fvec

def all_features(contents):
    vectorlize = CountVectorizer()
    sents,sents1,sents2,scores,train_vec= [],[],[],[],[]
    for line in contents[:-1]:
        sents1.append(line[1])
        sents2.append(line[2])
        sents.append(line[1])
        sents.append(line[2])
        if len(line) > 3:
            scores.append(float(line[3]))
        train_vec.append(np.array(feature_vector(line[1].split(' ')[:-1], line[2].split(' ')[:-1])))
    Sents = vectorlize.fit_transform(sents).toarray()
    Sents1 = vectorlize.transform(sents1).toarray()
    Sents2 = vectorlize.transform(sents2).toarray()
    with open("model.pickle", "wb") as f:
        pickle.dump(vectorlize, f)
    tfidf_Sents1, tfidf_Sents2, tfidf_Sents= [],[],[]
    cosine,pos_lcs,glove_ws = [],[],[]
    model = models.KeyedVectors.load_word2vec_format("glove.txt")
    for i in range(len(sents1)):
        tfidf_Sent1, tfidf_Sent2 = tf_idf(Sents1[i], Sents2[i])
        tfidf_Sents1.append(tfidf_Sent1)
        tfidf_Sents2.append(tfidf_Sent2)
        tfidf_Sents.append(tfidf_Sent1)
        tfidf_Sents.append(tfidf_Sent2)
        cosine.append(cosine_similarity([tfidf_Sent1], [tfidf_Sent2])[0][0])
        lcs1, lcs2 = tag_and_parser(sents1[i], sents2[i])
        pos_lcs.append([lcs1, lcs2])
        glove_ws.append(model.wmdistance(sents1[i].split(' ')[:-1], sents2[i].split(' ')[:-1]))
    tp_Sents = topic_id(tfidf_Sents)
    featrures = np.c_[glove_ws, cosine, pos_lcs, tp_Sents[::2], tp_Sents[1::2], train_vec]
    return featrures,scores



def extract():
    """
    通过scaler进行了向量的归一化处理，取值都在[0,1]之间
    另外增加了无监督学习和MT blue的预测值作为特征。
    此处的train_Blue对应MT blue,train_score 对应Unsupervised Learning
    :return:
    """
    scaler = preprocessing.MinMaxScaler()
    content1 = loadfile('train.txt')
    blue_train = loadfile("train_BLUE.txt")
    train_b = []
    for line in blue_train[:-1]:
        train_b.append(float(line[1]))
    train_b = np.array(train_b)
    score_train = loadfile("train_score.txt")
    train_s = []
    for line in score_train[:-1]:
        train_s.append(float(line[1]))
    train_s = np.array(train_s)
    train_feature ,scores= all_features(content1)
    train_feature = scaler.fit_transform(train_feature)
    X_train = np.c_[train_feature,train_b,train_s]
    X_train = scaler.fit_transform(X_train)
    Y_train = np.array(scores)

    content2 = loadfile("test.txt")
    blue_test = loadfile("test_BLUE.txt")
    test_b = []
    for line in blue_test[:-1]:
        test_b.append(float(line[1]))
    test_b = np.array(test_b)
    score_test = loadfile("test_score.txt")
    test_s = []
    for line in score_test[:-1]:
        test_s.append(float(line[0].split(' ')[1]))
    test_s = np.array(test_s)
    Y_ids = []
    for line in content2:
        if len(line) < 2:
            break
        Y_ids.append(line[0])
    test_featreus, scores = all_features(content2)
    X_test = np.c_[test_featreus,test_b,test_s]
    X_test = scaler.transform(X_test)
    with open("data.pickle", "wb") as f:
        pickle.dump([X_train, Y_train, X_test, Y_ids], f)
    return X_train, Y_train, X_test, Y_ids

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_ids = extract()