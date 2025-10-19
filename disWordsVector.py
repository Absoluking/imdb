import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
#nltk.download()
#导入内置的日志记录模块并进行配置 以便Word2Vec能生成友好的输出信息
import logging
from gensim.models import word2vec

train = pd.read_csv("labeledTrainData.tsv", delimiter="\t",quoting=3,header=0)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
#print("Read %d labeled train reviews,%d labeled test reviews, "\
#      "%d unlabeled reviews\n"%(train["review"].size,\
#        test["review"].size,unlabeled_train["review"].size))#验证是否成功读入

def review_to_wordlist(review,remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()#去除html标签
    review_text = re.sub("a-zA-Z"," ",review_text)#去除非字母
    words = review_text.lower().split()#将文本转成小写并拆分
    if remove_stopwords:#默认为False
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


#下载punkt分词器
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#拆分句子
def review_to_sentences(review,tokenizer,remove_stopwords=False):
    #使用分词器将段落拆分成句子
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        #句子为空则跳过
        if len(raw_sentence)>0:
            #将单词列表加入句子列表中
            sentences.append(review_to_wordlist(raw_sentence,remove_stopwords))
    #返回句子列表（每一个句子是一个单词列表）
    return sentences

sentences = []
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review,tokenizer)
print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
#print(sentences[0])#test
#print(sentences[1])#test
print(len(sentences))

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',\
        level=logging.INFO)
#设置参数
num_features = 300#字向量维度：更多的特征会导致更长的运行时间
min_word_count = 40#最小字数：这有助于将词汇表的大小限制为有意义的单词
num_workers = 6#工作线程：要运行的并行进程数
context = 10#上下文/窗口大小：训练算法应考虑多少个上下文单词
downsampling = 1e-3#常用字词的下采样: Google 文档建议的值介于 .00001 和 .001 之间

print("Training model...")
model = word2vec.Word2Vec(sentences,workers=num_workers,\
        vector_size = num_features,min_count=min_word_count,\
        window=context,sample=downsampling)

#保存模型
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)


