import pickle


class Model:
    """
    Class to represent a model
    """
    def __init__(self, model_path) -> None:
        # Path ..models/xgb_model_cifar10_300_seed_17_val_accuracy.pkl
        self.model_path = model_path
        self.model = self.load_model(model_path)
        
        attributes = model_path.split('/')[-1].split('_')
        self.dataset = attributes[2]
        self.data_size = int(attributes[3])
        self.seed = int(attributes[5])
        self.metric = attributes[6] + '_' + (attributes[7].split('.')[0])
        
    @staticmethod
    def load_model(model_path):
        """
        Load a model from a pickle file.

        Args:
            model_path (str): Path to the pickle file

        Returns:
            _type_: _description_
        """
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
        return model
    
