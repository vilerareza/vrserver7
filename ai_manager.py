from asyncio.format_helpers import extract_stack
import datetime
import base64
import pickle

from tensorflow.keras import models
import numpy as np
#from openvino.inference_engine import IECore

from cv2 import imread, resize, rectangle, imdecode, imencode
import numpy as np

from faceobject.models import FaceObject
from logs.models import Log, FrameLog

#aiManager = None

class AI_Manager():

    detector1 = None
    detector2 = None
    classifier = None
    classes = None
    classPrimaryKeys = None
    classVectors = None
    modelLocation = '' #"E:/testimages/facetest/vggface/ir/saved_model.xml"
    ieModelProperties = []
    recognitionThreshold = 0.2

    def __init__(self, recognition = False, ie = False, model_location = '', classes_location = ''):
        
        # Face detector 1 (haarcascade)
        from cv2 import CascadeClassifier
        self.detector1 = CascadeClassifier("vision_AI/model/haarcascade_frontalface_default.xml")
        # Face detector 2 (mtcnn)
        from mtcnn.mtcnn import MTCNN
        self.detector2 = MTCNN()

        # Classifier
        if recognition:
            self.modelLocation = model_location
            if model_location != '':
                if ie:
                    pass
                    # Use intel inference engine
                    # self.classifier = self.create_inference_engine(self.modelLocation)
                else:
                    # Use regular tf / keras model
                    self.classifier = models.load_model(self.modelLocation)
            else:
                print ('Model location is not set')
    
        # Classes
        self.classPrimaryKeys, self.classVectors = self.get_class_objects(model = FaceObject)

    # def create_inference_engine(self, model_location):
    #     ie = IECore()
    #     net  = ie.read_network(model = model_location)
    #     input_name = next(iter(net.input_info))
    #     output_name = next(iter(net.outputs))
    #     self.ieModelProperties = input_name, output_name
    #     try:
    #         model = ie.load_network(network = self.ieModelLocation, device_name = "MYRIAD")
    #         print ("USE NCS2 VPU")
    #     except:
    #         model = ie.load_network(network = self.ieModelLocation, device_name = "CPU")
    #         print ("NCS2 not found, use CPU...")
        
    #     return model
    
    def bound_faces(self, detector_type, bytes_data):
        # Get image byte data, detect face and create bounding box, return image byte data.
        try:
            # Conversion to np array
            buff = np.asarray(bytearray(bytes_data))
            img = imdecode(buff, 1)
            # Check type of detector
            if detector_type == 1:
                # Haarcascade detector perform here
                detector = self.detector1
                bboxes = detector.detectMultiScale(img)
                img = self.draw_rect(img, bboxes)
            elif detector_type == 2:
                # MTCNN detector perform here
                detector = self.detector2
                detection = detector.detect_faces(img)
                bboxes = []
                for dict in detection:
                    bboxes.append(dict['box'])
                img = self.draw_rect(img, bboxes)
            # Returning bytes data
            _, img_bytes = imencode(".jpg", img)
            return img_bytes.tobytes()
        except Exception as e:
            print (f'bound_face: {e}')
            return bytes_data

    def detect_faces(self, detector_type, img):
        # Detect faces and return bounding boxes
        # Check type of detector
        if detector_type == 1:
            # Haarcascade detector perform here
            detector = self.detector1  
            bboxes = detector.detectMultiScale(img)
        elif detector_type == 2:
            # MTCNN detector perform here
            detector = self.detector2
            detection = detector.detect_faces(img)
            bboxes = []
            for dict in detection:
                bboxes.append(dict['box'])
        return bboxes

    def extract_faces(self, detector_type, bytes_data, target_size = (224,224), draw_bbox = False):
        # Detect faces and return face images
        # Conversion to np array
        try:
            buff = np.asarray(bytearray(bytes_data))
            img = imdecode(buff, 1)
            # Detect face and get bounding boxes
            bboxes = self.detect_faces(detector_type, img)
            # Extract faces
            faces = []
            for box in bboxes:
                x1, y1, width, height = box
                x2, y2 = x1 + width, y1 + height
                # face data array
                face = img[y1:y2, x1:x2]
                # Resizing
                factor_y = target_size[0] / face.shape[0]
                factor_x = target_size[1] / face.shape[1]
                factor = min (factor_x, factor_y)
                face_resized = resize(face, (int(face.shape[0]* factor), int(face.shape[1]*factor)))
                diff_y = target_size[0] - face_resized.shape[0]
                diff_x = target_size[1] - face_resized.shape[1]
                # Padding
                face_resized = np.pad(face_resized,((diff_y//2, diff_y - diff_y//2), (diff_x//2, diff_x-diff_x//2), (0,0)), 'constant')
                faces.append(face_resized)
            return bboxes, faces, img
        except Exception as e:
            print (f'extract_faces: {e}')
            return [], [], []

    def extract_primary_face(self, detector_type, image_path, target_size = (224,224)):
        # Detection
        img, box = self.detect_primary_face_from_file(detector_type, image_path)
        if np.any(box):
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height
            # face data array
            face = img[y1:y2, x1:x2]
            # Resizing
            factor_y = target_size[0] / face.shape[0]
            factor_x = target_size[1] / face.shape[1]
            factor = min (factor_x, factor_y)
            face_resized = resize(face, (int(face.shape[0]* factor), int(face.shape[1]*factor)))
            diff_y = target_size[0] - face_resized.shape[0]
            diff_x = target_size[1] - face_resized.shape[1]
            # Padding
            face_resized = np.pad(face_resized,((diff_y//2, diff_y - diff_y//2), (diff_x//2, diff_x-diff_x//2), (0,0)), 'constant')
            # Progress
            return face_resized
        return None

    def detect_primary_face_from_file(self, detector_type, image_path):
        # Check type of detector
        if detector_type == 1:
            detector = self.detector1
        elif detector_type == 2:
            detector = self.detector2

        img = imread(image_path)

        if detector == self.detector1:
            # Haarcascade detector perform here
            img = imread(image_path)
            bboxes = detector.detectMultiScale(img)
            if (len(bboxes)>0):
                # Face detected
                print ('Detector 1: Face detected')
                box = bboxes[0]
                return img, box
            else:
                return img, []

        elif detector == self.detector2:
            # Haarcascade detector perform here
            detection = detector.detect_faces(img)
            if (len(detection)>0):
                print ('Detector 2: Face detected')
                box = detection[0]['box']
                return img, box
            else:
                return img, []

    def create_face_vectors(self, face_list):
        # Create face vectors numpy array from list of face data
        try: 
            face_vectors = []
            for face in face_list.copy():
                face = np.expand_dims(face, axis=0)
                face = face/255
                # Predict vector
                vector = self.classifier.predict(face)[0]
                face_vectors.append(vector)
            face_vectors = np.array(face_vectors)
            return face_vectors
        except Exception as e:
            print (f'Create face vector:{e}')
            return []

    def create_mean_face_vector(self, face_list):
        # Create mean face vector numpy array from list of face data
        if self.classifier:
            face_vector = []
            for face in face_list.copy():
                face = np.expand_dims(face, axis=0)
                face = face/255
                #face = np.moveaxis(face, -1, 1)
                print (f'face shape: {face.shape}')
                # Predict vector
                vector = self.classifier.predict(face)[0]
                face_vector.append(vector)
            face_vector = np.array(face_vector)
            face_vector = np.mean(face_vector, axis = 0)
            print (f'face_vector shape: {face_vector.shape}')
            return face_vector
        else:
            print ('No classifier, mean face vector not created')
            return []

    def make_classifier(self, ie = False, model_location = ''):
        self.modelLocation = model_location
        if self.modelLocation != '':
            if ie:
                pass
                # Use intel inference engine
                # self.classifier = self.create_inference_engine(self.modelLocation)
            else:
                # Use regular tf / keras model
                self.classifier = models.load_model(self.modelLocation)
            return True
        else:
            print ('Model location is not set')
            return False

    def draw_rect(self, img, boxes):
        try:
            for box in boxes:
                xb, yb, widthb, heightb = box
                rectangle(img, (xb, yb), (xb+widthb, yb+heightb), color = (232,164,0), thickness = 3)
            return img
        except Exception as e:
            return img

    def get_class_objects(self, model):
        '''
        Get the class objects fro database
        Returns:
        1. List of class primary keys
        2. List of class vectors
        '''
        primaryKeys = []
        vectors = []
        try:
            faceObjects = model.objects.all()
            for faceObject in faceObjects:
                primaryKeys.append(faceObject.pk)
                vectors.append(pickle.loads(base64.b64decode(faceObject.faceVector)))
        except Exception as e:
            print (f'get_class_obects: {e}')
        finally:
            return primaryKeys, vectors

    def find_distance_to_classes_euc(self, sample_vector, class_vectors):
        # Return list of distance from sample vector to every vectors in db_vectors list
        distances = []
        for vector in class_vectors:
            # Euclidean distance
            distance = sum(np.power((sample_vector - vector), 2))
            distance = np.sqrt(distance)
            distances.append(distance)
        return distances

    def find_distance_to_classes_cos(self, sample_vector, class_vectors):
        # Return list of distance from sample vector to every vectors in db_vectors list
        distances = []
        for vector in class_vectors:
            # Cosine distance
            dot = np.dot(sample_vector, vector)
            norm = np.linalg.norm (sample_vector) * np.linalg.norm (vector)
            similarity = dot / norm
            distances.append(1-similarity)
        return distances

    def find_closest_distance(self, distances):
        # Find the smallest value in distance and return the value and its index
        nearest = np.min(distances)
        index = np.argmin(distances)
        return nearest, index

    def find_vector_key(self, vectors, class_vectors, class_keys):
        '''
        Find nearest class for each element in vectors
        Returns:
        1. List of nearest class id
        2. List of 
        '''
        keys = []
        nearestDists = []
        if len(class_keys) != len(class_vectors):
            print ('find_vector_key: class_vectors and class_keys does not match')
            return keys, nearestDists
        for vector in vectors:
            # Calculate distance between vector and every element in classes vectors
            distances = self.find_distance_to_classes_cos(vector, class_vectors)
            #print (f'DISTANCES: {distances}')
            # Find closest distance and its index
            nearestDist, index = self.find_closest_distance(distances)
            id = class_keys[index]
            keys.append(id)
            nearestDists.append(nearestDist)
        return keys, nearestDists

    def log_frame(self, frame_bytes):
        frameLog = FrameLog.objects.create(
            frameData = base64.b64encode(frame_bytes).decode('ascii')
        )
        frameLog.save()
        return frameLog

    def log_face(self, time_stamp, face_key, face_data, frame_id, bbox):
        # Retrieve face object (fo) with id = face_key
        #fo = FaceObject.objects.get(id=face_key)
        log = Log.objects.create(
            objectID = face_key, 
            timeStamp = time_stamp,
            faceData = base64.b64encode(pickle.dumps(face_data)).decode('ascii'),
            frameID = frame_id,
            bbox = base64.b64encode(pickle.dumps(bbox)).decode('ascii')
        )
        log.save()

    def recognize(self, detector_type, bytes_data, target_size = (224,224), draw_bbox = False):
        timeStamp = datetime.datetime.now()
        #bboxes, faces, img_bytes = self.extract_faces(detector_type, bytes_data, target_size, draw_bbox)
        bboxes, faces, imgNp = self.extract_faces(detector_type, bytes_data, target_size, draw_bbox)
        if len(faces)>0:
            # Logging the frame
            frameLog = self.log_frame(bytes_data)
            # Recognize the face and log the face
            i = 0
            for i in range (len(faces)):
                faceVector = self.create_face_vectors([faces[i]])
                faceKey, faceNearestDist = self.find_vector_key(faceVector, self.classVectors, self.classPrimaryKeys)
                if faceNearestDist[0] < self.recognitionThreshold:
                    self.log_face(timeStamp, faceKey[0], faces[i], frameLog.id, bbox = bboxes[i])
            
            # If bounding box is enabled
            if draw_bbox:
                # Create image with bounding boxes
                self.draw_rect(imgNp, bboxes)
                # Convert back the img to BytesIO and return the bytes
                _, imgNp = imencode(".jpg", imgNp)
                return imgNp.tobytes()

        return bytes_data

try:
    aiManager = AI_Manager(recognition = True, model_location = "vision_AI/model/vgg_model_loaded.h5")
    print ('model created')
except Exception as e:
    print (f'Error on activating Vision AI: {e}')
