from gensim.models import Word2Vec
model = Word2Vec.load('300features_40minwords_10context')#使用已经训练好的模型
#print(model.wv.doesnt_match("man woman child kitchen".split()))
#print(model.wv.doesnt_match("france england germany berlin".split()))
#print(model.wv.doesnt_match("paris berlin london austria".split()))
#print(model.wv.most_similar("man"))
#print(model.wv.most_similar("queen"))
#print(model.wv.most_similar("awful"))
#print(model)




