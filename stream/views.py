
from django.http import JsonResponse
#from .consumers import frames
from .consumers import deviceIPs

# def get_stream_status(request, device_name):
#     stream = False
#     if device_name in frames:
#         stream = True
#     statusResponse = JsonResponse({'stream': stream})
#     return statusResponse

def get_device_ip(request, device_name):
    if device_name in deviceIPs:
        statusResponse = JsonResponse({'ip': deviceIPs[device_name]})
    else:
        statusResponse = JsonResponse({'ip': None})
    return statusResponse