from utils.args import list_of_param_dicts

class Exp_Basic(object):
    def __init__(self, param_dict, gpus=0):
        """
        Basic experiment class            
        """
        self.params = list_of_param_dicts(param_dict)
        self.gpus = gpus
        self.device = None
        self.model = None
        self.best_args = None
    
    def _build_model(self, args):
        raise NotImplementedError
    
    def _get_data_loader(self, data_path):
        raise NotImplementedError
    
    def _get_logger(self, args):
        raise NotImplementedError
    
    def train(self):
        pass

    def retrain(self):
        pass

    def test(self):
        pass