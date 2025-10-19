from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import time
import disWordsVector as dv
train = pd.read_csv("labeledTrainData.tsv", delimiter="\t",quoting=3,header=0)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
#数据清洗
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(dv.review_to_wordlist(review,remove_stopwords=True))
print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(dv.review_to_wordlist(review,remove_stopwords=True))

start = time.time()
model = Word2Vec.load("300features_40minwords_10context")
word_vectors = model.wv.vectors
#设置k为词汇数量的五分之一
num_clusters = word_vectors.shape[0]//5
#初始化一个k-means对象并用它提取中心
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)
end = time.time()
elapsed = end-start
print("执行kmeans聚类花费了",elapsed,"秒")
#当前每个单词的聚类分配结果存储在'idx'中，
# 而原始Word2Vec模型中的词汇表仍保存在'model.wv.index_to_key'中。
# 为方便使用，我们通过以下方式将两者压缩为一个字典
word_centroid_map = dict(zip(model.wv.index_to_key,idx))
for cluster in range(10):
    print("\nCluster %d" % cluster)
    words = []
    #找属于当前聚类的所有单词
    for i in range(len(word_centroid_map.values())):
        if(list(word_centroid_map.values())[i]==cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)
#将评论转换为质心袋
def create_bag_of_centroids(wordlist,word_centroid_map):
    #聚类的数量等于最高的聚类索引加一
    num_centroids = max(word_centroid_map.values())+1
    #预分配质心袋向量
    bag_of_centroids = np.zeros(num_centroids,dtype="float32")
    #找单词属于哪一个聚类中心 并将该聚类中心计数加一
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index]+=1
    return bag_of_centroids
#为训练集分配质心袋
train_centroids = np.zeros((train["review"].size,num_clusters),dtype="float32")
counter = 0
#调用函数将训练集转换为质心袋
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review,word_centroid_map)
    counter+=1
#为测试集分配质心袋
test_centroids = np.zeros((test["review"].size,num_clusters),dtype="float32")
counter = 0
#调用函数将测试集转换为质心袋
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review,word_centroid_map) 
    counter+=1
#预测
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)
output = pd.DataFrame(data={"id":test["id"],"sentiment":result})
output.to_csv("BagOfCentroids.csv",index=False,quoting=3)

