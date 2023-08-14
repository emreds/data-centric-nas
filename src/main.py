import copy
import random

from naslib.search_spaces.nasbench101.graph import is_valid_edge, is_valid_vertex
import logging

import pickle
import sys
import time

from naslib.search_spaces import (
    SimpleCellSearchSpace,
    NasBench101SearchSpace,
    HierarchicalSearchSpace,
)
from naslib.search_spaces.nasbench101 import graph

from naslib.utils import get_dataset_api

from naslib.search_spaces.nasbench101.encodings import EncodingType
from naslib.search_spaces.nasbench101 import encodings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler

from schema import ArchCoupled
import surrogate_front
import neighborhood
from collections import deque
import plot


# I am creating class based on architecture because I want to keep many data for every pass.
class ParetoLocalSearch:
    def __init__(self, arch, iterations, dataset_api) -> None:
        self.base_arch = arch
        self.iterations = iterations
        self.dataset_api = dataset_api
        self.archive = [arch]
        self.iter_queue = deque(self.archive)
        self.arch_hashes = set([self.base_arch.hash])
        self.trained_arch_cnt = 0
        self.history = {0: [arch]}
    
    
    def search(self):
        i = 0 
        
        while i < self.iterations:
            if len(self.iter_queue) > 0: 
                step_arch = self.iter_queue.popleft()
                nbhd = neighborhood.get_nbhd(step_arch.spec, self.dataset_api)
                self.trained_arch_cnt += len(nbhd)
                refined_nbhd = self._find_non_dominated_solutions(nbhd + [step_arch])
                
                for nei in refined_nbhd: 
                    for ex_arch in self.archive: 
                        if nei != ex_arch and self._is_dominated(ex_arch, nei):
                            self.archive.remove(ex_arch)
                            self.arch_hashes.remove(ex_arch.hash)
                            if ex_arch in self.iter_queue:
                                self.iter_queue.remove(ex_arch)
                
                for nei in refined_nbhd:
                    if nei.hash not in self.arch_hashes:
                        self.arch_hashes.add(nei.hash)
                        self.archive.append(nei)
                        self.iter_queue.append(nei)

            print(f"This is archive length for step {i+1}: {len(self.archive)}")
        
            i += 1
            
            self.history[i] = [copy.deepcopy(arch) for arch in self.archive]
            
        print(f"Total trained architectures: {self.trained_arch_cnt}")
        
        pareto_front = self._find_non_dominated_solutions(self.archive)

        return pareto_front
    
    
    def _find_non_dominated_solutions(self, pareto_front):
        non_dominated_solutions = []

        for sol1 in pareto_front:
            is_dominated_by_others = False
            for sol2 in pareto_front:
                if sol1 != sol2 and self._is_dominated(sol2, sol1):
                    is_dominated_by_others = True
                    break

            if not is_dominated_by_others:
                non_dominated_solutions.append(sol1)

        return non_dominated_solutions
    
    
    @staticmethod
    def _is_dominated(sol1, sol2):
        return sol1.val_accuracy >= sol2.val_accuracy and sol1.train_time <= sol2.train_time



if __name__ == "__main__":
    dataset_api = get_dataset_api("nasbench101", "cifar10")
    train_time_model_path = "/p/project/hai_nasb_eo/emre/data_centric/NASLib/naslib/runners/predictors/xgb_model_cifar10_1000_seed_12_train_time.pkl"
    acc_model_path = "/p/project/hai_nasb_eo/emre/data_centric/NASLib/naslib/runners/predictors/xgb_model_cifar10_1000_seed_12_val_accuracy.pkl"

    
    # Load the XGBoost model from the pickle file
    with open(train_time_model_path, 'rb') as file:
        time_model = pickle.load(file)
        
    with open(acc_model_path, 'rb') as file:
        acc_model = pickle.load(file)
        
    # Get the surrogate front
    surrogate_front = surrogate_front.get_surrogate_front(dataset_api, acc_model, time_model)
    
    paretos = []
    for arch in surrogate_front:
        starting_arch = ArchCoupled(arch, dataset_api["nb101_data"])
        PLS = ParetoLocalSearch(starting_arch, 100, dataset_api)
        pareto_ls = PLS.search()
        paretos += pareto_ls
    
    full_front = PLS._find_non_dominated_solutions(paretos)
    plot.plot_pareto_front(full_front, path="pareto_front_pls.png")

    
# First get the non dominated neighbours + current arch.
# Then, compare them with the archive.
# Remove the dominated ones from the archive.
# Add the non dominated ones to the archive and iter_queue.
