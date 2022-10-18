from django.db import models
from faceobject.models import FaceObject
# Create your models here.

class FrameLog(models.Model):
    frameData = models.CharField(max_length = 250000)

class Log(models.Model):
    #objectID = models.ForeignKey(FaceObject, on_delete=models.CASCADE, blank=True, null = True)
    #frameID = models.ForeignKey(FrameLog, on_delete=models.CASCADE, blank=True, null = True)
    objectID = models.IntegerField(null = True)
    timeStamp = models.DateTimeField(max_length = 30)
    faceData = models.CharField(max_length = 250000)
    frameID = models.IntegerField(null = True)
    bbox = models.CharField(max_length = 255, null = True)

    def __str__(self):
        return (f'{self.objectID}-{self.timeStamp}')


