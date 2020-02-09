from django.shortcuts import render
from .models import file_upload
from django.views.generic import ListView, CreateView
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import os
from .post_file import PostForm # new
from django.urls import reverse_lazy # new
import sys
sys.path.append("file/")
import nlp
from .models import file_upload


# Create your views here.
class UploadPageView(ListView):
    model = file_upload
    template_name = 'upload.html'

class CreatePostView(CreateView): # new
    model = file_upload
    form_class = PostForm
    template_name = 'post_file.html'
    success_url = '/read/'

#def add_file(request):
topic_dist = None
topics_per_word = None
phis = None
topic_words = None
def analyze(text):
    global topic_dist
    global topics_per_word
    global phis
    global topic_words
    grams = nlp.preprocess(text)
    assert(len(grams) > 0)
    nlp.mass_vectorize(grams)
    assert(len(nlp.vectors) > 0)
    assert(nlp.corpus != None and len(nlp.corpus) > 0)
    #tops = file_upload.num
    nlp.train_lda()
    topic_dist, topics_per_word, phis, topic_words = nlp.do_lda(grams)
    paras = nlp.split_paragraphs(text)
    bow_grams = []
    assert(len(grams) == len(paras))
    for i in range(len(grams)):
        bow_grams.append([gram for sentence in grams[i] for gram in sentence])
    word_list = []
    for i in range(len(paras)):
        word_list.append(nlp.stable_matching(paras[i], bow_grams[i], topics_per_word))
    return [word for para in word_list for word in para]

def read_file(request):
    #p = "/Users/administrator/Documents/GitHub/LittleBirdie/little_birdie/uscsite/media/text/test.txt"
    folder = "media/text"
    file_content = "hello"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        f = open(file_path, 'r', errors = 'ignore')
        file_content = f.read()
        f.close()
    #"../media/text/test.txt"
    #print(file_upload.num)
    word_list = analyze(file_content)
    template = loader.get_template('upload.html')
    topic_colors = []
    topic_rep = []
    cnt = 0
    curr_topic = topic_words[0]
    for i in range(nlp.NUM_TOPICS):
    #    assert(curr_topic[0] >= topic_words[max(cnt, len(topic_words)-1)])
        curr_topic = topic_words[cnt]

        if(i == curr_topic[0]):
            topic_colors.append(nlp.color_key[i])
            if len(curr_topic[1]) > 0:
                topic_rep.append(str(curr_topic[1]))
            else:
                topic_rep.append(None)
            cnt+=1

        else:
            topic_colors.append(nlp.color_key[-1])
            topic_rep.append(None)
    #topic_colors = [nlp.color_key[topic[0]] for topic in topic_words]
    #topic_rep = [word[1] for word in topic_words]
    context = {
        'file_content': file_content,
        'word_list': word_list,
        'topic_dist': topic_dist,
        'topics_per_word': topics_per_word,
        'topic_words': topic_rep,
        'topic_colors': topic_colors,
        'num_topics': list(range(nlp.NUM_TOPICS))
        }
    return HttpResponse(template.render(context))
