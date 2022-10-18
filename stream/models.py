from django.db import models
from threading import Condition

# Create your models here.

class FrameObject(models.Model):
    content = bytes()
    condition = Condition()
    name = models.CharField(max_length=100)

    def __str__(self):
        return (self.name)