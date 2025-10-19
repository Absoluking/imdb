#Porter可以词干提取和词形还原
import pandas as pd  #提供读写数据文件功能
from bs4 import BeautifulSoup
import numpy as np
import re#处理正则表达式的包
#import nltk
#nltk.download()#下载停用词表
from sklearn.ensemble import RandomForestClassifier#随机森林分类器
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords #使用nltk来获取停用词表（a,the,and...）
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
#header=0表示文件的第一行包含列名 delimiter="\t"表示字段由制表符分隔，quoting=3代表忽略双引号
train.shape
train.columns.values
example1 = BeautifulSoup(train["review"][0]) #删除html标签
#print(train["review"][0])#列名为review的第一行数据
#print()
#print(example1.get_text())#获取清洗后的文本
letters_only = re.sub("[^a-zA-Z]"," ",example1.get_text())#将example1文本中非字母替换为空格
lower_case = letters_only.lower()#转换为小写
words = lower_case.split()#拆分成单独的单词  NLP标记化
#print(lower_case)
#print(stopwords.words("english"))#显示英文停用词表
#print()
#words = [w for w in words if not w in stopwords.words("english")]#删除words中的停用词
#print(words)
def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()#去除html标签
    letters_only = re.sub("[^a-zA-Z]", " ",review_text)#去除非字母
    words = letters_only.lower().split()#转换为小写并拆分成单独的单词
    stops = set(stopwords.words("english"))#创建停用词集合
    meaningful_words = [w for w in words if not w in stops]#去除停用词
    return " ".join(meaningful_words)#合并为一个段落
#clean_review = review_to_words(train["review"][0])#测试函数
#print(clean_review)
num_reviews = train["review"].size
clean_train_review = []
for i in range(num_reviews):
    if (i+1)%1000 ==0:
        print("Review %d of %d\n" % (i+1,num_reviews))
    clean_train_review.append(review_to_words(train["review"][i]))
    #CountVectorizer带有自己的选项自动执行标记化 预处理 停用词删除
vectorizer = CountVectorizer(analyzer="word",\
                             tokenizer=None,\
                            preprocessor=None,\
                            stop_words=None,\
                            max_features=5000)#使用最常见的5000个单词  
train_data_features = vectorizer.fit_transform(clean_train_review)#拟合模型
train_data_features = train_data_features.toarray()
print(train_data_features.shape)#25000行和5000个功能（每个词汇一个）
vocab = vectorizer.get_feature_names_out()#查看词汇表
#print(vocab)
#dist = np.sum(train_data_features,axis=0)#每个单词的计数
#for tag,count in zip(vocab,dist):#打印每个单词出现在训练集的次数
#    print(count,tag)
print("Training the random forest...")
forest = RandomForestClassifier(n_estimators=100)#初始化随机森林树数量为100
forest = forest.fit(train_data_features,train["sentiment"])#x=词袋，y=sentiment
#测试集
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
print(test.shape)#25000行2列
num_reviews = len(test["review"])
clean_test_reviews = []
print("Cleaning and parsing the test set movie reviews...\n")
for i in range(num_reviews):
    if (i+1)%1000 == 0:
        print("Review %d of %d\n" % (i+1,num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)#使用随机森林预测结果
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("Bag_of_Words_model.csv", index=False,quoting=3)



