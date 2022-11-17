from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from .serializers import FaceSerializer
from .models import FaceObject
import ai_manager

# Create your views here.
class FaceList (generics.ListCreateAPIView):
    queryset = FaceObject.objects.all()
    serializer_class = FaceSerializer

    aiManager = ai_manager.aiManager

    def post(self, request, *args, **kwargs):
        exist = FaceObject.objects.filter(faceID = request.data['faceID'])  # exist -> queryset object
        if exist:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        else:
            resp = self.create(request, *args, **kwargs)
            print (resp)
            try:
                with self.aiManager.condition:
                    self.aiManager.get_class_objects(model = FaceObject)
            except Exception as e:
                print (e)
            return resp


class FaceDetail (generics.RetrieveUpdateDestroyAPIView):
    queryset = FaceObject.objects.all()
    serializer_class = FaceSerializer