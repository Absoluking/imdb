from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import disWordsVector as dv
model = Word2Vec.load("300features_40minwords_10context")
print(type(model.wv.vectors))
print(model.wv.vectors.shape)#行数代表模型词汇表中的字数 列数对应于特征向量的大小
print(model.wv["flower"])#1*300 numpy数组
#段落中所有单词向量取平均
def makeFeatureVec(words,model,num_features):
    #初始化
    featureVec = np.zeros((num_features),dtype="float32")
    nwords = 0
    #一个包含模型词汇表中单词名称的列表
    index2word_set = set(model.wv.index_to_key)
    #循环遍历评论中的单词 如果单词在模型中则将它的特征向量加到总体上
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec,model.wv[word])
    #总特征向量除以单词数得到向量平均
    featureVec = np.divide(featureVec,nwords)
    return featureVec
#计算每一个评论的向量平均
def getAvgFeatureVecs(reviews,model,num_features):
    #初始化计数器
    counter = 0
    #声明二维数组用来存储所有评论的向量平均
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #循环遍历评论
    for review in reviews:
        if counter%1000==0:
            print("Review %d of %d"%(counter,len(reviews)))
        #调用取向量平均函数
        reviewFeatureVecs[counter] = makeFeatureVec(review,model,num_features)
        counter = counter+1
    return reviewFeatureVecs
num_features = 300
train = pd.read_csv("labeledTrainData.tsv", delimiter="\t",quoting=3,header=0)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(dv.review_to_wordlist(review,remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews,model,num_features)
print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(dv.review_to_wordlist(review,remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews,model,num_features)

forest = RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...")
forest.fit(trainDataVecs,train["sentiment"])
result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("Word2Vec_AverageVectors.csv",index=False,quoting=3)



