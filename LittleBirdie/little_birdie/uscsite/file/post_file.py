# posts/forms.py
from django import forms
from .models import file_upload

class PostForm(forms.ModelForm):

    class Meta:
        model = file_upload
        fields = ['text']
