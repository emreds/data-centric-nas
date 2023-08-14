import numpy as np
from scipy.spatial import distance
from scipy.spatial.qhull import ConvexHull

def hypervolume(pareto_front, reference_point):
    pareto_front = np.array(pareto_front)
    reference_point = np.array(reference_point)
    front_size = pareto_front.shape[0]
    
    # Calculate the hypervolume using the reference point
    volume = np.prod(np.maximum(pareto_front - reference_point, 0), axis=1).sum()
    return volume

def spread(pareto_front):
    pareto_front = np.array(pareto_front)
    convex_hull = ConvexHull(pareto_front)
    hull_volume = convex_hull.volume
    front_size = pareto_front.shape[0]
    return hull_volume / front_size

def generational_distance(pareto_front1, pareto_front2):
    pareto_front1 = np.array(pareto_front1)
    pareto_front2 = np.array(pareto_front2)
    
    distances = []
    for p1 in pareto_front1:
        min_distance = min([distance.euclidean(p1, p2) for p2 in pareto_front2])
        distances.append(min_distance)
    
    avg_distance = sum(distances) / len(distances)
    return avg_distance

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
