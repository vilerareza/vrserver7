from threading import Condition, Lock

class Frame(object):
    def __init__(self) -> None:
        self.condition = Condition(lock = Lock())
        self.content = bytes()

class Control(object):
    def __init__(self) -> None:
        self.condition = Condition(lock = Lock())
        self.command = None