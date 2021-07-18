import yaml

class ConfigClass:
    def __init__(self,filename):
        self.filename =  filename

    def Loading_Config(self):
        with open(self.filename,"r") as f:
            data = yaml.safe_load(f)
        return data

