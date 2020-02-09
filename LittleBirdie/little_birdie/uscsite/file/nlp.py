import gensim as gs
import numpy as np
import os
from gensim.test.utils import common_corpus

stopwords = frozenset({"ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"})
gowords = frozenset({"y u no work"})
model = gs.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
lda_model = None
id2word = None
corpus = None

NUM_TOPICS = 100
vectors = []
# file = open("test.txt")
# content = file.read()

print("done loading model")
def split_paragraphs(raw_text):
    paras = raw_text.splitlines()
    paras = [p.strip() for p in paras]
    paras = [p for p in paras if p != ""]
    #paras = [p + '.' for p in paras]
    return paras


def sim_tokenize(raw_sentence):
    return list(gs.utils.simple_tokenize(raw_sentence.lower()))

def remove_stopwords(raw_sentence):
    cooked = []
    for word in raw_sentence:
        if not word.lower() in stopwords and not word.isnumeric():
            cooked.append(word)
    return cooked

# pars = split_paragraphs(content)
def tokenize(raw_para):#return  list of strings
    stemmer = gs.parsing.porter.PorterStemmer()
    sentences = gs.summarization.textcleaner.split_sentences(raw_para) # list of sentences
    #sentences = [sentence.strip() for sentence in sentences if sentence.strip() != ""]
    sentences = [sentence for sentence in sentences if sentence != []]
    unpunctual = map(gs.parsing.preprocessing.strip_punctuation, sentences)
    split_sent = list(map(lambda x: remove_stopwords(x.split()), sentences))#list(sentences) of list(words)
    stemmed = []
    for i in range(len(split_sent)):
        stemmed.append([stemmer.stem(word) for word in split_sent[i]])
    #lemmatized = [gs.utils.lemmatize(s,stopwords = stopwords) for s in sentences]# lemmatizes and tokenizes, return list of list of tokens

    return stemmed

def cons_tokenize(raw_para):
    stemmer = gs.parsing.porter.PorterStemmer()
    sentences = gs.summarization.textcleaner.split_sentences(raw_para) # list of sentences
    sentences = [sentence for sentence in sentences if sentence != []]
    unpunctual = map(gs.parsing.preprocessing.strip_punctuation, sentences)
    split_sent = list(map(lambda x: x.split(), sentences))#list(sentences) of word strings
    stemmed = [stemmer.stem(word) for sentence in split_sent for word in sentence]
    #lemmatized = [stemmer.stem_sentence(s) for s in sentences]# lemmatizes and tokenizes, return list of list of tokens
    return stemmed

def preprocess(raw_text):
    paras = split_paragraphs(raw_text)#list of paragraphs
    tokens = [tokenize(p) for p in paras] #list (paragraphs) of list(sentences) of list (per word) of tokens
    phrases = gs.models.phrases.Phrases(sentences = [sentence for paragraph in tokens for sentence in paragraph], delimiter = b'_', common_terms = stopwords)
    gram = [[]]
    for i in range(len(tokens)):
        gram.append([])
        gram[i].extend([phrases[s] for s in tokens[i]])# list(sentences) of list(per word) of bigrams
    gram = list(filter(lambda x: x!=[], gram))
    return gram #list (paragraphs) of list(sentences) of list (per word) of bigrams

def vectorize(word):
    if word in model:
        return model[word]
    else:
        return np.zeros((300,))

def gs_vectorize(grams):
    flattened = []
    for paragraph in grams:
        flattened.append([gram for sentence in paragraph for gram in sentence])
    id2word = gs.corpora.Dictionary(flattened)
    # Create Corpus
    texts = flattened
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return corpus, id2word

text8 = None
gs_corpus = ""
gs_id2word = {}
def get_gs():
    global text8
    global gs_corpus
    global gs_id2word
    print(os.getcwd())
    text8 = open("../data/text8.txt", "r")
    print("read text8")
    gs_corpus = preprocess(text8.read())
    print("preprocessing")
    gs_corpus, gs_id2word = gs_vectorize(gs_corpus)
    print("done loading corpus")
print("gs")
get_gs()
def mass_vectorize(grams):
    global vectors
    global id2word
    global corpus
    flattened = []
    for paragraph in grams:
        flattened.append([gram for sentence in paragraph for gram in sentence])
    id2word = gs.corpora.Dictionary(flattened)
    # Create Corpus
    texts = flattened
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    vectors = []
    for i in range(len(grams)):
        paragraph = grams[i]
        vectors.append([])
        for j in range(len(paragraph)):
            sentence = paragraph[j]
            vectors[i].append([])
            for k in range(len(sentence)):
                word = sentence[k]
                vectors[i][j].append(vectorize(word))
print("gs_vectorize run, starting to train in")
lda_model = gs.models.LdaMulticore(gs_corpus, num_topics=NUM_TOPICS, workers = 4, id2word = gs_id2word)
print("finished training initial model")
def train_lda():
    global lda_model

    other_corpus = corpus
    lda_model.update(other_corpus)
    #lda_model = gs.models.LdaModel(corpus = corpus, id2word = id2word, num_topics = NUM_TOPICS, minimum_phi_value=0.33)

def kill_dims(vectors):
    return [word for paragraph in vectors for sentence in paragraph for word in sentence]

def do_lda(grams):

    doc_grams = [gram for paragraph in grams for sentence in paragraph for gram in sentence]
    corpus = id2word.doc2bow(doc_grams)
    topic_dist, topics_per_word, phis = lda_model.get_document_topics(bow = corpus, per_word_topics = True)
    return topic_dist, topics_per_word, phis, lda_model.print_topics(NUM_TOPICS)

def decompose_bigram(gram, topics_per_word):
    unigrams = gram.split('_')
    words = []
    bigram_id = id2word.token2id[gram]
    for unigram in unigrams:
        id = id2word.token2id.get(unigram, id2word.token2id[gram])
        if len(topics_per_word[id][1]) > 0:
            words.append(Word("null",unigram,id,topics_per_word[bigram_id][1][0]))
        else:
            words.append(Word("null",unigram,id,-1))
        #words.append(Word("null", unigram, id, topics_per_word[bigram_id][1][0]))
    return words


def stable_matching(raw_para, grams, topics_per_word):
    input = cons_tokenize(raw_para) #postlemma
    raw_tokens = raw_para.split()
    fixed_grams = []
    word_list = []
    for j in grams:
        if '_' in j:
            fixed_grams.extend(decompose_bigram(j,topics_per_word))
        else:
            id = id2word.token2id[j]
            if len(topics_per_word[id][1]) > 0:
                fixed_grams.append(Word("null",j,id,topics_per_word[id][1][0]))
            else:
                fixed_grams.append(Word("null",j,id,-1))
    print(len(fixed_grams), len(input))
    gram_counter = 0
    assert(len(raw_tokens)==len(input))
    for i in range(len(input)):
        if gram_counter < len(fixed_grams) and input[i].lower()==fixed_grams[gram_counter].postlemma.lower():
            word_list.append(Word(raw_tokens[i],input[i],fixed_grams[gram_counter].id,fixed_grams[gram_counter].topic))
            gram_counter+=1
        else:
            word_list.append(Word(raw_tokens[i],raw_tokens[i],None,-1))
    return word_list

def colorize():
    global color_key
    import random
    for i in range(8,NUM_TOPICS):
        color_key[i] = "rgb("+str(int(random.random()*128)+128)+","+str(int(random.random()*128)+128)+","+str(int(random.random()*128)+128)+")"

color_key = {-1: "rgba(0,0,0,0)",
            0: "#9dc6d8", #light blue
            1: "#e38690", #pink
            2: "#7dd0b6", #green
            3: "#d2b29b", #light brown
            4: "#f69256", #orange
            5: "#ccc1db", #lilac
            6: "#ead98b", #yellow
            7: "#b2b1a5"} #light gray
print("colorizing: ")
colorize()
class Word:
    def __init__(self, prelemma,postlemma, id, topic):
        self.prelemma = prelemma
        self.postlemma = postlemma
        self.id = id
        self.topic = topic
        self.color = color_key[topic]

    def __str__(self):
        return self.prelemma + "  :  " + self.postlemma + "  :  " + self.topic + "  :  " + self.color + ";;;\n"

    def __repr__(self):
        return self.prelemma + "  :  " + self.postlemma + "  :  " + str(self.topic) + "  :  " + str(self.color) + ";;;\n"
