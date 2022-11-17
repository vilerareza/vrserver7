from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from .serializers import FaceSerializer
from .models import FaceObject

# Create your views here.
class FaceList (generics.ListCreateAPIView):
    queryset = FaceObject.objects.all()
    serializer_class = FaceSerializer
    def post(self, request, *args, **kwargs):
        exist = FaceObject.objects.filter(faceID = request.data['faceID'])  # exist -> queryset object
        if exist:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        else:
            return self.create(request, *args, **kwargs)

class FaceDetail (generics.RetrieveUpdateDestroyAPIView):
    queryset = FaceObject.objects.all()
    serializer_class = FaceSerializer