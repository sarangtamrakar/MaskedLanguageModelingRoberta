from transformers import RobertaTokenizerFast
import torch
from log import LogClass
from Config import ConfigClass


class RobertClass:
    def __init__(self):
        # Loading Config Data
        self.configObj = ConfigClass("params.yaml")
        self.configData = self.configObj.Loading_Config()
        self.logFile = self.configData['LoggingFileName']
        self.ModelName = self.configData['Loading']['Model_name']
        self.TokenizerName = self.configData['Loading']['TokenizerDir']
        self.top_k = self.configData['Prediction']['top_k']

        self.loggerObj = LogClass(self.logFile)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loggerObj.Logger("Device : " + str(self.device))
        self.loggerObj.Logger("Model is Loading")
        # Model & Tokenizer Loading..
        self.Model = torch.load(self.ModelName)
        self.loggerObj.Logger("Model Loaded")
        self.loggerObj.Logger(self.TokenizerName)
        self.tokenizer = RobertaTokenizerFast.from_pretrained("Tokenizer/")
        self.loggerObj.Logger("Tokenizer Loaded")

    def Encodings(self, text):
        try:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            mask_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
            self.loggerObj.Logger("Got Encoding & Mask_index")
            return input_ids, mask_index
        except Exception as e:
            self.loggerObj.Logger("Exception Occured in Encoding method of RobertClass : " + str(e))
            return "Exception Occured in Encoding method of RobertClass : " + str(e)

    def Prediction(self, ctext):
        try:
            if "[MASK]" not in str(ctext):
                self.loggerObj.Logger("[MASK] token not added yet.")
                return "Please Add [MASK] token in sentence, then pass as input."

            # replacing mask str via maskToken
            text = str(ctext).replace("[MASK]", self.tokenizer.mask_token)
            # getting encoding
            input_ids, mask_index = self.Encodings(text)
            # passing input to Model
            token_logits = self.Model(input_ids).logits
            # selecting the tensor of mask_token probability
            mask_token_logits = token_logits[0, mask_index, :]
            # finding the topk mask token id
            top_5_tokens = torch.topk(mask_token_logits,self.top_k, dim=1).indices[0].tolist()

            result_lis = []
            for i in top_5_tokens: # here we are decoding the id to words
                result_lis.append(self.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True))

            # binding the prediction into dictionary format..
            dic = {"Top 5 Token": result_lis, "Original Text": str(ctext)}
            self.loggerObj.Logger("Get the Result " + str(dic))

            return dic
        except Exception as e:
            self.loggerObj.Logger("Exception Occured in Prediction method Of RobertClass : " + str(e))
            return "Exception Occured in Prediction method Of RobertClass : " + str(e)
