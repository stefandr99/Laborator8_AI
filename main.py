from gensim.models import Word2Vec
import nltk
import re


def preprocess_text(text_file):
    f = open("StopWords", "r")
    stop = f.read().split('\n')
    f2 = open(text_file)
    t = f2.read()
    sentences = nltk.sent_tokenize(t)
    words = list()
    for s in sentences:
        w = re.findall(re.compile('\w+'), s.lower())
        words.append(w)
    extracted_words = list()
    for s in words:
        extr_sent = [w for w in s if w not in stop]
        extracted_words.append(extr_sent)
    return extracted_words


def sentences_to_one_hot(text_file):
    sentences = preprocess_text(text_file)
    s_concat = [subword for word in sentences for subword in word]
    sentence_set_list = list(set(s_concat))
    V = len(sentence_set_list)
    words_dict = dict()
    for w in sentence_set_list:
        words_dict[w] = sentence_set_list.index(w)
    text_in_hot_words = list()
    for sen in sentences:
        hot_words_sentence = list()
        for w in sen:
            hot_word = [0 for _ in range(V)]
            hot_word[words_dict[w]] = 1
            hot_words_sentence.append(hot_word)
        text_in_hot_words.append(hot_words_sentence)
    print(text_in_hot_words)


def learning(text_file):
    sentences = preprocess_text(text_file)
    model = Word2Vec(sentences, min_count=1, sg=1, window=3)
    return model


def get_similarities(file, words_list):
    model = learning(file)
    for w in words_list:
        print(f'{w} -> {model.wv.most_similar(w)[:5]}')


if __name__ == '__main__':
    get_similarities("text3", ['peace', 'king', 'faithful', 'ship', 'palace'])
    #sentences_to_one_hot('text')
    #print(preprocess_text('text'))

