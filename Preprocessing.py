#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-01-30 10:28
# @Author  : Caviar_liu
# @Site    : 
# @File    : Preprocessing.py
# @Software: PyCharm
# @Contact : zhiqiang.liu@gmail.com

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

def remove_stopwords(word_list):
    """
    :param word_list: 去除
    :return:
    """
    stop_words = stopwords.words('english')
    filter_words = [w for w in word_list if not w in stop_words]
    #stopWords =  [w for w in word_list if w in stop_words]
    #stopWords = " ".join(stopWords)
    # with open("stopwords.txt", "a") as f:
    #     f.write(stopWords)
    return filter_words

def wordnet(sentence, stopwords = True):
    """
    :param sentence: 单个句子
    :param stopwords:
    :return: 词性还原以及去除了stopwords的sentence
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(word, 'n').lower() for word in sentence.split()]
    words = [wordnet_lemmatizer.lemmatize(word, 'a').lower() for word in words]
    words = [wordnet_lemmatizer.lemmatize(word, 'v').lower() for word in words]
    translation_table = str.maketrans(string.punctuation + string.ascii_uppercase + string.digits,
                                      " " * len(string.punctuation) + string.ascii_lowercase + " " * len(string.digits))
    if stopwords:
        words = remove_stopwords(words)
    words = [word for word in words]
    new_sent =  ' '.join(words)
    new_sent = new_sent.translate(translation_table)
    return new_sent


def loadfile(filename):
    """
    :param filename: 文件的路径
    :return: [[id,senten1,senten2,score],....]
    """
    new_lines = []
    with open(filename, "r") as f:
        content = f.read()
        lines = content.split("\n")
        for line in lines:
            new_lines.append(line.split("\t"))
    return new_lines


def lines_processing(lines, stopwords = True):
    """
    :param lines:
    :param stopwords: 确定是否去停用词
    :return: 将处理好的text存入文件
    """
    new_content = ""
    line_len = len(lines[0]) > 3
    for line in lines:
        sen1 = wordnet(line[1], stopwords)
        sen2 = wordnet(line[2], stopwords)
        if line_len:
            line = line[0] + "\t" + sen1 + "\t" + sen2 + "\t" + line[3] + "\n"
            # print(line)

        else:
            line = line[0] + "\t" + sen1 + "\t" + sen2 + "\n"
        new_content += line
    if line_len:
        with open("train.txt","w") as f:
            f.write(new_content)
    else:
        with open("test.txt","w") as f:
            f.write(new_content)

if __name__ == '__main__':
    train_text = loadfile("train_ai-lab.txt")
    lines_processing(lines = train_text,stopwords = True)
    test_text = loadfile("test_ai-lab.txt")
    lines_processing(test_text,True)
