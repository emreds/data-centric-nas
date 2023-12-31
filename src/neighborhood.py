import copy
import random

import numpy as np
from naslib.search_spaces.nasbench101.graph import (is_valid_edge,
                                                    is_valid_vertex)
from schema import ArchCoupled

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
INPUT = "input"
OUTPUT = "output"


def format_neighbors(neighbors, dataset_api):
    valid_neighbours = []

    for arch in neighbors: 
        if (arch.ops is not None) and (len(arch.ops) == NUM_VERTICES):
            arch.hash = arch.hash_spec(OPS)
            if arch.hash in dataset_api["nb101_data"].fixed_statistics:
                coupled_arch = ArchCoupled(arch.hash, dataset_api=dataset_api["nb101_data"])
                valid_neighbours.append(coupled_arch)
        
    return valid_neighbours
        

def get_nbhd(spec, dataset_api: dict) -> list:
    # return all neighbors of the architecture
    matrix, ops = spec["module_adjacency"], spec["module_operations"]
    nbhd = []

    if matrix.shape[0] < NUM_VERTICES: 
        return []

    # add op neighbors
    for vertex in range(1, OP_SPOTS + 1):
        if is_valid_vertex(matrix, vertex):
            available = [op for op in OPS if op != ops[vertex]]
            for op in available:
                new_matrix = copy.deepcopy(matrix)
                new_ops = copy.deepcopy(ops)
                new_ops[vertex] = op
                model_spec = dataset_api["api"].ModelSpec(new_matrix, new_ops)
                #model_spec.hash = model_spec.hash_spec(OPS)
                nbhd.append(model_spec)

    # add edge neighbors
    for src in range(0, NUM_VERTICES - 1):
        for dst in range(src + 1, NUM_VERTICES):
            new_matrix = copy.deepcopy(matrix)
            new_ops = copy.deepcopy(ops)
            new_matrix[src][dst] = 1 - new_matrix[src][dst]

            if matrix[src][dst] and is_valid_edge(matrix, (src, dst)):
                model_spec = dataset_api["api"].ModelSpec(new_matrix, new_ops)
                #model_spec.hash = model_spec.hash_spec(OPS)
                nbhd.append(model_spec)

            if not matrix[src][dst] and is_valid_edge(new_matrix, (src, dst)):
                model_spec = dataset_api["api"].ModelSpec(new_matrix, new_ops)
                #model_spec.hash = model_spec.hash_spec(OPS)
                nbhd.append(model_spec)
    
    #random.shuffle(nbhd)
    
    
    
    return format_neighbors(nbhd, dataset_api)

def sample_random_architecture(self, dataset_api: dict, seed: int) -> None:
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        From the NASBench repository:
        one-hot adjacency matrix
        draw [0,1] for each slot in the adjacency matrix
        """
        
        np.random.seed(seed)
        
        while True:
            matrix = np.random.choice([0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT

            spec = dataset_api["api"].ModelSpec(matrix=matrix, ops=ops)
            if dataset_api["nb101_data"].is_valid(spec):
                break

        set_spec({"matrix": matrix, "ops": ops})