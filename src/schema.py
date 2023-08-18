
class ArchCoupled: 
    """
    Class to represent an architecture coupled with its performance
    """
    def __init__(self, hash_value, dataset_api, epoch=108) -> None:
        stats = dataset_api.computed_statistics[hash_value][epoch][0]
        self.train_accuracy = stats['final_train_accuracy']
        self.val_accuracy = stats['final_validation_accuracy']
        self.test_accuracy = stats['final_test_accuracy']
        self.train_time = stats['final_training_time']
        
        self.hash = hash_value
        
        fixed_stats = dataset_api.fixed_statistics[hash_value]
        self.module_adjacency = fixed_stats['module_adjacency']
        self.module_operations = fixed_stats['module_operations']
        self.trainable_parameters = fixed_stats['trainable_parameters']
        
        self.spec = {'module_adjacency': self.module_adjacency, 'module_operations': self.module_operations}

    
    def __json__(self): 
        return {
            "train_accuracy": self.train_accuracy, 
            "validation_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy, 
            "train_time": self.train_time, 
            "hash": self.hash, 
            "spec": {'module_adjacency': self.module_adjacency.tolist(), 'module_operations': self.module_operations}

        }