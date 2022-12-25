from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.response import Response

from .models import Log, FrameLog
from .serializers import LogFrameSerializer, LogIDSerializer, LogSerializer

import datetime

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

class LogListDateFilter(generics.ListAPIView):
    
    def get_queryset(self, id, num, date_gte, date_lte):
        # If the log with requested attribute value does not exist, the following will just return 0
        
        # Conversion to date object
        if date_gte != '0':
            date_gte = datetime.datetime.strptime(date_gte, '%d%m%y%H%M')
            if date_lte != '0':
                # Data range between date_gte and date_lte
                date_lte = datetime.datetime.strptime(date_lte, '%d%m%y%H%M')
                # Return it, based on the number of data requested
                if num != -1:
                    return Log.objects.filter(objectID = id, timeStamp__range = [date_gte, date_lte])[:num]
                else:
                    return Log.objects.filter(objectID = id, timeStamp__range = [date_gte, date_lte])
            else:
                # Data with date later than date_gte
                # Return it, based on the number of data requested
                if num != -1:
                    return Log.objects.filter(objectID = id, timeStamp__gte = date_gte)[:num]
                else:
                    return Log.objects.filter(objectID = id, timeStamp__gte = date_gte)
        
        elif date_lte != '0':
            # Data with date later than date_gte
            # Return it, based on the number of data requested
            date_gte = datetime.datetime.strptime(date_gte, '%d%m%y%H%M')
            if num != -1:
                return Log.objects.filter(objectID = id, timeStamp__lte = date_lte)[:num]
            else:
                return Log.objects.filter(objectID = id, timeStamp__lte = date_lte)
        
        else:
            # All data, no date is specified
            if num != -1:
                return Log.objects.filter(objectID = id)[:num]
            else:
                return Log.objects.filter(objectID = id)


    def list(self, request, id=None, num=-1, date_gte='', date_lte=''):
        #queryset = self.get_queryset(id, num, date_gte, date_lte)
        queryset = Log.objects.all()
        serializer = LogSerializer(queryset, many = True)
        return Response(serializer.data)