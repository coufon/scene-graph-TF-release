class Query:
    def __init__(self, frame, msg=None):
        self.frame = frame
        self.msg = msg

class DetectResult:
    def __init__(self, bb_list):
        self.bb_list = bb_list
