from rest_framework import serializers
from .models import Log, FrameLog

class LogSerializer(serializers.ModelSerializer):
    class Meta:
        model = Log
        fields= ['id', 'objectID', 'timeStamp', 'faceData', 'frameID', 'bbox']

class LogIDSerializer(serializers.ModelSerializer):
    class Meta:
        model = Log
        fields= ['objectID']

class LogFrameSerializer(serializers.ModelSerializer):
    class Meta:
        model = FrameLog
        fields= ['id','frameData']