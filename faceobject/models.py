from django.db import models

# Create your models here.

class FaceObject(models.Model):
    faceID = models.CharField(max_length=20)
    firstName = models.CharField(max_length = 30, blank=True)
    lastName = models.CharField(max_length = 30, blank=True)
    faceVector = models.CharField(max_length = 20000, blank=True)
    faceData = models.CharField(max_length = 2500000, blank=True)

    def __str__(self):
        return (f'{self.faceID}-{self.firstName}')

        