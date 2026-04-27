from uuid import uuid4
from datetime import datetime


class RunState:
    def __init__(self):
        self.run_id = str(uuid4())
        self.started_at = datetime.now()
        self.steps = {}
        self.errors = []
        self.metadata = {}


    def mark(self, name, status, detail=None):
        self.steps[name] = {
            "status": status,
            "detail": detail,
        }


    def fail(self, error):
        self.errors.append(str(error))

