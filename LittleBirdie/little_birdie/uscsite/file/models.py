from django.db import models
from django.conf import settings
import os, shutil

class file_upload(models.Model):
    folder = 'media/text'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    text = models.FileField(upload_to='text/')
    #num = models.TextField(default=3)


# Create your models here.
