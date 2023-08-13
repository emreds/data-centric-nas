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


def pareto_local_search(arch, iterations, dataset_api):
        
        archive = [arch]
        iter_queue = deque(archive)
        arch_hash = arch.hash
        arch_hashes = set([arch_hash])
        #nbhd = get_nbhd()
        i = 0 
        
        while i < iterations:
            if len(archive) > 0: 
                last_arch = iter_queue.popleft()
                nbhd = neighborhood.get_nbhd(last_arch.spec, dataset_api)
                print(f"This is nbhd: {nbhd}")
                for nei in nbhd:
                    if nei.hash not in arch_hashes:
                        dominated_archs = []
                        #print(f"This is nei: {nei}")
                        #print(f"This is archive length: {len(archive)}")
                        for ex_arch in archive:
                            #print(f"This is ex_arch_val_acc: {ex_arch.val_accuracy}")
                            #print(f"This is nei_val_acc: {nei.val_accuracy}")
                            is_dominated = False
                            if nei.hash not in arch_hashes:
                                if nei.val_accuracy < ex_arch.val_accuracy and nei.train_time > ex_arch.train_time:
                                    # Marks the neighbor as dominated
                                    is_dominated = True
                                    
                                if nei.val_accuracy > ex_arch.val_accuracy and nei.train_time < ex_arch.train_time:
                                    # Finds the archs dominated by the neighbor.
                                    dominated_archs.append(ex_arch)
                            
                        if not is_dominated:
                                arch_hashes.add(nei.hash)
                                archive.append(nei)
                                iter_queue.append(nei)
                                # If the neighbor is non dominated, removes the dominated archs from the archive.
                                for dom_arch in dominated_archs:
                                    if dom_arch in archive:
                                        print("removing")
                                        archive.remove(dom_arch)
                                        arch_hashes.remove(dom_arch.hash)
                if len(archive) == 1:
                    break
            print(f"This is archive length for step {i}: {len(archive)}")
            i += 1
            
        return archive



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
    
    starting_arch = ArchCoupled(surrogate_front[15], dataset_api["nb101_data"])
    outputs = pareto_local_search(starting_arch, 10, dataset_api)
    print(outputs)