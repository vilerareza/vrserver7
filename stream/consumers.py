from channels.generic.websocket import WebsocketConsumer
from threading import Thread
from functools import partial
from .streamobject import Frame
import ai_manager
import time

# global frames dict containing frames objects from existing deviceframeConsumer objects 
frames = {}
deviceIPs = {}
aiManager = ai_manager.aiManager


class DeviceFrameConsumer(WebsocketConsumer):

    global frames
    global deviceIPs
    
    stop_flag = False
    processThisFrame = True
    t_process_frame = Thread()
    
    def connect(self):
        # Extract device info
        self.device_name = self.scope['url_route']['kwargs']['device_name']
        self.device_ip = self.scope['client'][0]
        try:
            # Register device IP to deviceIPs dict
            deviceIPs[self.device_name] = self.device_ip
            # Create frame object and register to frames dict
            self.frame = Frame()
            frames[self.device_name] = self.frame
        except:
            pass

        self.accept()
        print (f'Device connected {self.device_name}')
        self.send(bytes_data = b'1')

    def disconnect(self, code):
        super().disconnect(code)
        try:
            deviceIPs.pop(self.device_name)
            frames.pop(self.device_name)
        except:
            pass
        self.stop_flag = True
        self.close()
        print (f'Device disconnected {self.device_name}')
        
    def receive(self, text_data=None, bytes_data=None):
        pass
        #print (f'Receive: Text: {text_data}')
        print (f'Len: {len(bytes_data)}')
        # Receive frame bytes data from camera
        # print (f'Receive: Text: {type(text_data)} Bytes: {len(bytes_data)}')
        if not self.t_process_frame.is_alive():
            self.t_process_frame = Thread(target = partial(self.start_process_frame, bytes_data))
            self.t_process_frame.daemon = True
            self.t_process_frame.start()
        else:
            print ('bypass')

    def start_monitor_stream(self):
        # Monitor the stream activity from device
        # Declare timeout and pop the self.frame object from frames dict
        def monitor_stream():
            while not self.stop_flag:
                try:
                    with self.frame.condition:
                        if not (self.frame.condition.wait(timeout = 10)):
                            print ('timeout')
                            self.disconnect(None)
                except Exception as e:
                    print (e)
        Thread(target=monitor_stream).start()

    def start_process_frame(self, bytes_data):
        # '''Detect and extraction'''
        #img_bytes = aiManager.bound_faces(detector_type = 1, bytes_data = bytes_data)
        '''Comment to disable the AI'''
        t1 = time.time()
        img_bytes = aiManager.recognize(detector_type = 1, bytes_data = bytes_data)
        t2 = time.time()
        print (f'time = {t1-t2}')
        # with self.frame.condition:
        #     self.frame.content = bytes_data
        #     #self.frame.content = img_bytes
        #     self.frame.condition.notify_all()
        # Request new frame from camera
        self.send(bytes_data = b'1')