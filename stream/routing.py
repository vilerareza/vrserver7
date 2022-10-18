from django.urls import re_path
from django.urls import path

from .consumers import DeviceFrameConsumer

websocket_urlpatterns = [
    re_path('ws/device/(?P<device_name>\w+)/$', DeviceFrameConsumer.as_asgi())
    ]