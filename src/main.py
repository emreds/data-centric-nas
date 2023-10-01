import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import metrics
import model
import neighborhood
import numpy as np
import plot
import surrogate_front as sf
from naslib.utils import get_dataset_api
from pls import ParetoLocalSearch
from schema import ArchCoupled

STARTING_POINTS = 20

def write_pareto_metrics(pareto_metrics: metrics.ParetoMetrics, path: Path) -> None:
    
    json_file_path = path / "pareto_metrics.json"
    
    with open(json_file_path, 'w') as file: 
        json.dump(pareto_metrics.__dict__, file, indent=4)
    
    pass

def write_pls_history(pls: ParetoLocalSearch, path: Path, start_point_id: int) -> None:
    
    history_file_path = path / (str(start_point_id) + "_history.json")
    history = pls.history
    history["trained_arch_cnt"] = pls.trained_arch_cnt
    
    with open(history_file_path, 'w') as file: 
        json.dump(history, file, default=lambda o: o.__json__(), indent=4)
    
    pass


def multi_surrogate_PLS(size_models: Dict,
                   dataset_api: Dict,
                   max_train_cnt:int,
                   starting_points:int,
                   min_max_dict: Dict,
                   result_dir: Path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis")
                   ) -> None:
    
    for size in size_models:
        surrogate_PLS(size_models=size_models,
                      dataset_api=dataset_api,
                      size=size,
                      max_train_cnt=max_train_cnt,
                      starting_points=starting_points,
                      min_max_dict=min_max_dict,
                      result_dir=result_dir)
        
    pass
    
    
def surrogate_PLS(size_models: Dict,
             dataset_api: Dict,
             size:int,
             max_train_cnt:int,
             starting_points:int,
             min_max_dict: Dict,
             result_dir: Path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis")
             ) -> None:
    
    
    for seed in size_models[size]:
        # Get the surrogate front
        surrogate_front = sf.get_surrogate_front(dataset_api, size_models[size][seed]["val_accuracy"].model, size_models[size][seed]["train_time"].model)
        np.random.seed(seed)
        if len(surrogate_front) >= starting_points:
            surrogate_front = np.random.choice(surrogate_front, size=starting_points, replace=False)
        else: 
            print(f"Surrogate front size is smaller than starting points. For the size: {size} and seed: {seed} Surrogate front size: {len(surrogate_front)}")
        
        search_res_dir = result_dir / str(seed)
        if not os.path.exists(search_res_dir):
            os.makedirs(search_res_dir)
        if max_train_cnt == 0:
            surrogate_front = [ArchCoupled(arch, dataset_api["nb101_data"]) for arch in surrogate_front]
            #formatted_front = neighborhood.format_neighbors(surrogate_front, dataset_api["nb101_data"])
            pareto_metrics = metrics.ParetoMetrics(surrogate_front, min_max=min_max_dict)
    
            write_pareto_metrics(pareto_metrics, path=search_res_dir)
            plot.plot_pareto_front(pareto_front=surrogate_front, min_max=min_max_dict, path=search_res_dir / "pareto_front.png")
        else:
            print(f"Making surrogate MO for size {size} and seed {seed} and len(surrogate_front): {len(surrogate_front)}")
            make_PLS(search_res_dir=search_res_dir,
                    starting_archs=surrogate_front,
                    max_train_cnt=max_train_cnt,
                    min_max_dict=min_max_dict,
                    )
        
    pass
    

def make_PLS(search_res_dir: Path, starting_archs: List[str], max_train_cnt: int, min_max_dict: Dict) -> int:
    
    paretos = []
    #min_max_dict = metrics.get_min_max_values(dataset_api["nb101_data"])
    trained_arch_cnt = 0
    pls_multi = {}
    
    for i, arch in enumerate(starting_archs):
        starting_arch = ArchCoupled(arch, dataset_api["nb101_data"])
        pls_multi[i] = ParetoLocalSearch(starting_arch, 1, dataset_api)
    
    while trained_arch_cnt < max_train_cnt:
        for i, arch in enumerate(starting_archs):
            pareto_ls = pls_multi[i].search()
            paretos += pareto_ls
        
        trained_arch_cnt = sum([pls_multi[i].trained_arch_cnt for i in range(len(starting_archs))])
    
    for i in range(len(starting_archs)):
        write_pls_history(pls_multi[i], search_res_dir, i+1)
        
    full_front = pls_multi[i]._find_non_dominated_solutions(paretos)
    
    pareto_metrics = metrics.ParetoMetrics(full_front, min_max=min_max_dict)
    
    write_pareto_metrics(pareto_metrics, path=search_res_dir)
    plot.plot_pareto_front(pareto_front=full_front, min_max=min_max_dict, path=search_res_dir / "pareto_front.png")
    
    return trained_arch_cnt
    
def run_surrogate_PLS(
    max_train_cnt:int, 
    starting_points:int,
    min_max_dict: Dict,
    model_dir: str = "../surrogates/models",
    result_dir:Path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis/surrogates")
    ):


    size_models = get_size_models(model_dir=model_dir)
    
    multi_surrogate_PLS(size_models=size_models,
                        dataset_api=dataset_api,
                        result_dir=result_dir,
                        max_train_cnt=max_train_cnt,
                        min_max_dict=min_max_dict,
                        starting_points=starting_points
                        )


def get_size_models(model_dir: Path) -> Dict:
    model_paths = os.listdir(model_dir)
    model_paths.sort()
    model_paths = [os.path.join(model_dir, file) for file in model_paths]
    
    size_models = defaultdict(lambda: defaultdict(dict))
    for model_path in model_paths:
        surr_model = model.Model(model_path)
        size_models[surr_model.data_size][surr_model.seed][surr_model.metric] = surr_model
        
    return size_models

def raw_MO(dataset_api: Dict, random_seed: int, min_max_dict: Dict, starting_points: int, max_train_cnt: int, search_res_dir:Path) -> None:
        np.random.seed(seed=random_seed)
        raw_MO_archs = np.random.choice(list(dataset_api["nb101_data"].fixed_statistics.keys()), size=starting_points, replace=False)
        #print(f"Raw MO archs: {raw_MO_archs}")
        print(f"Making raw MO for seed {random_seed} and starting points {starting_points}")
        if not os.path.exists(search_res_dir):
            os.makedirs(search_res_dir)
        make_PLS(
            search_res_dir=search_res_dir,
            starting_archs=raw_MO_archs,
            max_train_cnt=max_train_cnt,
            min_max_dict=min_max_dict,
            )
        
        pass

def multi_raw_MO_PLS(dataset_api: Dict, min_max_dict: Dict, random_seeds: List[int], starting_points: int, max_train_cnt: int, result_dir:Path) -> None:
    
    for seed in random_seeds:
        search_res_dir = result_dir / str(seed)
        raw_MO(dataset_api=dataset_api,
               random_seed=seed,
               min_max_dict=min_max_dict,
               starting_points=starting_points,
               max_train_cnt=max_train_cnt,
               search_res_dir=search_res_dir
            )
    
    pass
    

if __name__ == "__main__":
    dataset_api = get_dataset_api("nasbench101", "cifar10")
    min_max_dict=metrics.get_min_max_values(dataset_api["nb101_data"])
    

    surr_train_cnt = 2000
    surrogate_mo_result_path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis/surrogates") / "30_runs" / str(surr_train_cnt)
    run_surrogate_PLS(
        max_train_cnt=surr_train_cnt,
        starting_points=STARTING_POINTS,
        min_max_dict=min_max_dict,
        result_dir=surrogate_mo_result_path,
        model_dir=Path("../surrogates/models/30_runs")
        )
    
  
    random_seeds = list(range(10, 310, 10))
    raw_mo_train_cnt = 2000
    raw_mo_result_path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis/raw_mo") / "30_runs" / str(raw_mo_train_cnt)
    multi_raw_MO_PLS(dataset_api=dataset_api,
                    min_max_dict=min_max_dict,
                    random_seeds=random_seeds,
                    starting_points=STARTING_POINTS,
                    max_train_cnt=raw_mo_train_cnt,
                    result_dir=raw_mo_result_path
                    )