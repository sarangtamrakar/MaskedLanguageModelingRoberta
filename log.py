from datetime import datetime


class LogClass:
    def __init__(self, file_name):
        self.file_name = file_name

    def Logger(self, message):
        try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.time = self.now.strftime("%H:%M:%S")

            forma = "Date : {}, Time : {}, Message : {} \n".format(str(self.date), str(self.time), str(message))

            with open(self.file_name, "a+") as f:
                f.write(forma)
            f.close()

        except Exception as e:
            f.close()
            return "Exception Occured during the Logging"
