import numpy as np
from scipy.spatial import distance
from scipy.spatial.qhull import ConvexHull


class ParetoMetrics:
    def __init__(self, pareto_front, reference_point = [0,0]) -> None:
        self.hypervolume = self.get_hypervolume(pareto_front, reference_point)
        self.spread = self.get_spread(pareto_front)
        self.generational_dist = self.get_generational_distance(pareto_front)

    def get_hypervolume(pareto_front, reference_point):
        """
        Calculates the hypervolume of a Pareto front.

        Args:
            pareto_front: A list of points in the Pareto front.
            reference_point: A point that represents the ideal solution.

        Returns:
            The hypervolume of the Pareto front.
        """

        pareto_front = np.array(pareto_front)
        reference_point = np.array(reference_point)

        volume = 0
        for i in range(len(pareto_front)):
            for j in range(i + 1, len(pareto_front)):
                if pareto_front[i] <= pareto_front[j]:
                    continue

                volume += np.prod(pareto_front[j] - pareto_front[i]) / np.prod(reference_point - pareto_front[i])

        return volume

    def get_spread(pareto_front):
        pareto_front = np.array(pareto_front)
        convex_hull = ConvexHull(pareto_front)
        hull_volume = convex_hull.volume
        front_size = pareto_front.shape[0]
        return hull_volume / front_size

    def get_generational_distance(pareto_front1, pareto_front2):
        pareto_front1 = np.array(pareto_front1)
        pareto_front2 = np.array(pareto_front2)
        
        distances = []
        for p1 in pareto_front1:
            min_distance = min([distance.euclidean(p1, p2) for p2 in pareto_front2])
            distances.append(min_distance)
        
        avg_distance = sum(distances) / len(distances)
        return avg_distance



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