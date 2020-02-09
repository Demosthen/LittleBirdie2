from django.urls import path
from .views import UploadPageView, CreatePostView
from .views import read_file

app_name = "file"

urlpatterns = [
    path('upload/', UploadPageView.as_view(), name="upload"),
    path('read/', read_file, name='read'),
    path('post/', CreatePostView.as_view(), name='add_post'),
    #path('add_file/',add_file, name = "add_file"),
]
