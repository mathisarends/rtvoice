class SubAgentDone(Exception):
    def __init__(self, result: str):
        self.result = result
