from rest_framework import serializers
from .models import FaceObject

class FaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceObject
        fields= ['id', 'faceID', 'firstName', 'lastName', 'faceVector', 'faceData']