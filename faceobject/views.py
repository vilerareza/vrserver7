from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from .serializers import FaceSerializer
from .models import FaceObject

# Create your views here.
class FaceList (generics.ListCreateAPIView):
    queryset = FaceObject.objects.all()
    serializer_class = FaceSerializer

class FaceDetail (generics.RetrieveUpdateDestroyAPIView):
    queryset = FaceObject.objects.all()
    serializer_class = FaceSerializer