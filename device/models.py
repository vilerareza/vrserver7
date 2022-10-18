from django.db import models

# Create your models here.
class Device(models.Model):
    #deviceID = models.AutoField(primary_key=True)
    deviceName = models.CharField(max_length=50)
    hostName = models.CharField(max_length = 50, default = 'hostname')
    wifiName = models.CharField(max_length=50, default = 'ssid')
    wifiPass = models.CharField(max_length=50, default= 'password')
    visionAI = models.BooleanField(default=False)

    def __str__(self):
        return self.deviceName
