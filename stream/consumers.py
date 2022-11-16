from channels.generic.websocket import WebsocketConsumer
from threading import Thread, Condition, Lock
from functools import partial
import ai_manager
import time

deviceFrameConsumers = {}
deviceIPs = {}
aiManager = ai_manager.aiManager

class DeviceFrameConsumer(WebsocketConsumer):

    global deviceFrameConsumers
    global deviceIPs
    
    stop_flag = False
    processThisFrame = True
    t_process_frame = Thread()
    condition = Condition(lock = Lock())
    
    def connect(self):
        # Extract device info
        self.device_name = self.scope['url_route']['kwargs']['device_name']
        self.device_ip = self.scope['client'][0]
        try:
            # Register self to consumers dict
            deviceFrameConsumers[self.device_name] = self
            # Register device IP to deviceIPs dict
            deviceIPs[self.device_name] = self.device_ip
        except:
            pass

        self.accept()
        print (f'Device connected {self.device_name}')
        self.send(bytes_data = b'1')

    def disconnect(self, code):
        super().disconnect(code)
        try:
            deviceFrameConsumers.pop(self.device_name)
            deviceIPs.pop(self.device_name)
        except:
            pass
        self.stop_flag = True
        self.close()
        print (f'Device disconnected {self.device_name}')
        
    def receive(self, text_data=None, bytes_data=None):
        # Receive frame bytes data from camera
        if not self.t_process_frame.is_alive():
            self.t_process_frame = Thread(target = partial(self.start_process_frame, bytes_data))
            self.t_process_frame.daemon = True
            self.t_process_frame.start()
        else:
            print ('bypass')

    def start_process_frame(self, bytes_data):
        # '''Detect and extraction'''
        #img_bytes = aiManager.bound_faces(detector_type = 1, bytes_data = bytes_data)
        '''Comment to disable the AI'''
        t1 = time.time()
        with self.condition:
            img_bytes = aiManager.recognize(detector_type = 1, bytes_data = bytes_data)
            self.condition.notify_all()
        t2 = time.time()
        print (f'time = {t2-t1}')
        self.send(bytes_data = b'1')