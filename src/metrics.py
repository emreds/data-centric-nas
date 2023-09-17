from typing import Dict, List

import numpy as np
from encode_arch import ArchitectureEncoder
from pymoo.indicators.hv import HV
from schema import ArchCoupled
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.spatial.qhull import ConvexHull


class ParetoMetrics:
    def __init__(self, pareto_front: List[ArchCoupled], min_max: Dict) -> None:
        """
        Useful class for calculating pareto metrics.

        Args:
            pareto_front (List[ArchCoupled]): _description_
            min_max (Dict): _description_
        """
        pareto_front_values = np.array(list(zip([arch.val_accuracy for arch in pareto_front], [arch.train_time for arch in pareto_front])))
        self.hypervolume = self.get_hypervolume(pareto_front_values, min_max)
        self.avg_hypervolume = self.hypervolume / len(pareto_front)
        self.diameter, self.podist_avg = self.get_diameter_podist_avg(pareto_front)
        self.best_acc = max([arch.val_accuracy for arch in pareto_front])
        self.best_train_time = min([arch.train_time for arch in pareto_front])        
        self.avg_acc = self.get_avg_acc(pareto_front)
        self.avg_train_time = self.get_avg_train_time(pareto_front)
        self.std_acc = self.get_std_acc(pareto_front)
        self.std_train_time = self.get_std_train_time(pareto_front)
        
    
    @staticmethod
    def get_spread(pareto_front):
        # Calculate the Euclidean distances between solutions
        distances = cdist(pareto_front, pareto_front, metric='euclidean')
        
        # Set diagonal elements (distance to itself) to a large value
        np.fill_diagonal(distances, np.inf)
        
        # Find the minimum distance for each solution
        min_distances = np.min(distances, axis=1)
        
        # Calculate the spread as the sum of minimum distances
        spread = np.sum(min_distances)
        
        # Normalize the spread by dividing by the number of solutions
        normalized_spread = spread / len(pareto_front)
        
        return normalized_spread
        
    @staticmethod
    def get_spread_old(pareto_front):
        pareto_front = np.array(pareto_front)
        convex_hull = ConvexHull(pareto_front)
        hull_volume = convex_hull.volume
        front_size = pareto_front.shape[0]
        return hull_volume / front_size
    
    @staticmethod
    def get_generational_distance(pareto_front1, pareto_front2):
        pareto_front1 = np.array(pareto_front1)
        pareto_front2 = np.array(pareto_front2)
        
        distances = []
        for p1 in pareto_front1:
            min_distance = min([distance.euclidean(p1, p2) for p2 in pareto_front2])
            distances.append(min_distance)
        
        avg_distance = sum(distances) / len(distances)
        return avg_distance
    
    @staticmethod
    def get_hypervolume(pareto_front: np.array, min_max_dict: dict) -> float:
        
        val_accuracy = pareto_front[:, 0]
        train_time = pareto_front[:, 1]
        
        norm_val_accuracy = normalize_values(values=val_accuracy, worst_value=min_max_dict['min_val_acc'], best_value=min_max_dict['max_val_acc'])
        # Shorter training time is better, so we invert the values.
        norm_train_time = scale_normalize(values=train_time, worst_value=min_max_dict['max_train_time'], best_value=min_max_dict['min_train_time'])
        ref_point = np.array([1, min_max_dict['max_train_time']])
        norm_pareto_front = np.array(list(zip(norm_val_accuracy, norm_train_time)))
        hv_calculator = HV(ref_point=ref_point)
        hv = hv_calculator.do(norm_pareto_front)
        
        return hv
        
    @staticmethod
    def get_avg_acc(pareto_front: List[ArchCoupled]) -> float:
        
        return sum([arch.val_accuracy for arch in pareto_front]) / len(pareto_front)
    
    @staticmethod
    def get_avg_train_time(pareto_front: List[ArchCoupled]) -> float:
        
        return sum([arch.train_time for arch in pareto_front]) / len(pareto_front)
    
    @staticmethod
    def get_std_acc(pareto_front: List[ArchCoupled]) -> float:
        
        return np.std([arch.val_accuracy for arch in pareto_front])
    
    @staticmethod
    def get_std_train_time(pareto_front: List[ArchCoupled]) -> float:
            
        return np.std([arch.train_time for arch in pareto_front])
    
    @staticmethod
    def get_diameter_podist_avg(pareto_archs: List[ArchCoupled]) -> (int, float):
        max_dist = 0 
        encoder = ArchitectureEncoder()
        total_dist = 0
        
        for i, arch_1 in enumerate(pareto_archs): 
            encoded_1 = encoder.encode_architecture(arch_1.module_adjacency, arch_1.module_operations)
            for j, arch_2 in enumerate(pareto_archs):
                if i == j:
                    continue
                encoded_2 = encoder.encode_architecture(arch_2.module_adjacency, arch_2.module_operations)
                xor_result = np.bitwise_xor(encoded_1, encoded_2)
                hamming_distance = np.count_nonzero(xor_result)

                if hamming_distance > max_dist:
                    max_dist = hamming_distance
                    total_dist += hamming_distance
        
        return max_dist, total_dist / len(pareto_archs)
        


def get_min_max_values(dataset_api, epoch=108):
    """
    Returns the reference point for the hypervolume calculation.
    
    Args:
        dataset_api: The dataset API object.
        epoch: The epoch to use for the reference point calculation.
        
    Returns:
        A dictionary containing the minimum and maximum values for validation accuracy and training time.
        
    """
    min_val_acc = float('inf')
    max_val_acc = 0
    min_train_time = float('inf')
    max_train_time = 0
    
    for hash_value in dataset_api.computed_statistics:
        stats = dataset_api.computed_statistics[hash_value][epoch][0]
        min_val_acc = min(stats['final_validation_accuracy'], min_val_acc)
        max_val_acc = max(stats['final_validation_accuracy'], max_val_acc)
        min_train_time = min(stats['final_training_time'], min_train_time)
        max_train_time = max(stats['final_training_time'], max_train_time)
        
    return {
        'min_val_acc': min_val_acc,
        'max_val_acc': max_val_acc,
        'min_train_time': min_train_time,
        'max_train_time': max_train_time
    }
        
def normalize_values(values: np.array, worst_value: float, best_value: float) -> np.array:
    """
    
    Normailizes the given values between worst_value and best_value.

    Args:
        values (np.array)
        worst_value (float)
        best_value (float)

    Returns:
        np.array
    """
    normalized_values = (values - worst_value) / (best_value - worst_value)
    return normalized_values


def scale_normalize(values: np.array, worst_value: float, best_value: float) -> np.array:
    """
    Scales and normalizes the given values between worst_value and best_value.

    Args:
        values (np.array)
        worst_value (float)
        best_value (float)

    Returns:
        np.array
    """
    
    normalized_values = normalize_values(values, worst_value, best_value)
    scaled_values = normalized_values * (worst_value - best_value) + best_value
    
    return scaled_values
    
    
    
'''
# Example usage
pareto_front1 = [
    [0.9, 5],
    [0.85, 8],
    [0.88, 6],
]

pareto_front2 = [
    [0.95, 4],
    [0.92, 7],
    [0.89, 5],
]

reference_point = [1, 0]
hv1 = hypervolume(pareto_front1, reference_point)
hv2 = hypervolume(pareto_front2, reference_point)

spread1 = spread(pareto_front1)
spread2 = spread(pareto_front2)

gd = generational_distance(pareto_front1, pareto_front2)

print(f"Hypervolume 1: {hv1}")
print(f"Hypervolume 2: {hv2}")
print(f"Spread 1: {spread1}")
print(f"Spread 2: {spread2}")
print(f"Generational Distance: {gd}")
'''