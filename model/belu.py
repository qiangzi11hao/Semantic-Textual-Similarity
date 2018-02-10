import nltk
with open('submimssioin_sample.txt','w',encoding='utf-8') as fin:
    with open('./data/test.txt',encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split('\t')
            str1=line[1]
            str2=line[2]
            BLEUscore = nltk.translate.bleu_score.corpus_bleu([str1],[str2])
            fin.write(line[0]+','+str(BLEUscore)+'\n')

BLEUscore = nltk.translate.bleu_score.corpus_bleu(['an apple','a pear'],['a apple','ll'])
print(BLEUscore)
