from gensim.models import word2vec

sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size = 200)
print(model['me'])
