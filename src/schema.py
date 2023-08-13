class ArchCoupled: 
    def __init__(self, hash_value, dataset_api, epoch=108) -> None:
        self.train_accuracy = dataset_api.computed_statistics[hash_value][epoch][0]['final_train_accuracy']
        self.val_accuracy = dataset_api.computed_statistics[hash_value][epoch][0]['final_validation_accuracy']
        self.test_accuracy = dataset_api.computed_statistics[hash_value][epoch][0]['final_test_accuracy']
        self.train_time = dataset_api.computed_statistics[hash_value][epoch][0]['final_training_time']
        self.hash = hash_value
        self.module_adjacency = dataset_api.fixed_statistics[hash_value]['module_adjacency']
        self.module_operations = dataset_api.fixed_statistics[hash_value]['module_operations']
        self.spec = {'module_adjacency': self.module_adjacency, 'module_operations': self.module_operations}
        self.trainable_parameters = dataset_api.fixed_statistics[hash_value]['trainable_parameters']