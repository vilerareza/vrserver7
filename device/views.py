from django.shortcuts import render
from rest_framework import generics
from .models import Device
from .serializers import DeviceSerializer
from rest_framework.response import Response
from rest_framework import status

# Create your views here.

class DeviceList (generics.ListCreateAPIView):
    queryset = Device.objects.all()
    serializer_class = DeviceSerializer

    def post(self, request, *args, **kwargs):
        exist = Device.objects.filter(deviceName = request.data['deviceName'])  # exist -> queryset object
        if exist:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        else:
            return self.create(request, *args, **kwargs)
    
class DeviceDetail (generics.RetrieveUpdateDestroyAPIView):
    queryset = Device.objects.all()
    serializer_class = DeviceSerializer