from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.response import Response

from .models import Log, FrameLog
from .serializers import LogFrameSerializer, LogIDSerializer, LogSerializer

class LogList(generics.ListAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer

class LogDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer

class LogListFaceID(generics.ListAPIView):
    queryset = Log.objects.all()
    serializer_class = LogIDSerializer

class LogListFaceIDFilter(generics.ListAPIView):
    def get_queryset(self, id=None):
        # If the log with the id does not exist, the following will just return 0
        return Log.objects.filter(objectID = id)

    def list(self, request, id):
        queryset = self.get_queryset(id)
        serializer = LogSerializer(queryset, many = True)
        return Response(serializer.data)

class FrameDetail(generics.RetrieveDestroyAPIView):
    queryset = FrameLog.objects.all()
    serializer_class = LogFrameSerializer