import copy
import json
import os
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List

import metrics
import model
import neighborhood
import numpy as np
import plot
import surrogate_front as sf
from naslib.utils import get_dataset_api
from schema import ArchCoupled

STARTING_POINTS = 20
PARETO_STEPS = 20


# I am creating class based on architecture because I want to keep many data for every pass.
class ParetoLocalSearch:
    def __init__(self, arch, max_train_cnt, dataset_api) -> None:
        self.base_arch = arch
        #self.iterations = iterations
        self.max_train_cnt = max_train_cnt
        self.dataset_api = dataset_api
        self.archive = [arch]
        self.iter_queue = deque(self.archive)
        self.arch_hashes = set([self.base_arch.hash])
        self.trained_arch_cnt = 0
        self.history = {0: [arch]}
    
    
    def search(self):
        step = 0 
        
        while self.trained_arch_cnt < self.max_train_cnt:
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

            #print(f"This is archive length for step {step+1}: {len(self.archive)}")
        
            step += 1
            
            self.history[step] = [copy.deepcopy(arch) for arch in self.archive]
            
        print(f"Total trained architectures: {self.trained_arch_cnt}")
        
        pareto_front = self._find_non_dominated_solutions(self.archive)

        return pareto_front
    
    
    def _find_non_dominated_solutions(self, pareto_front):
        non_dominated_solutions = []

        for sol1 in pareto_front:
            is_dominated_by_others = False
            for sol2 in pareto_front:
                if sol1 != sol2 and self._is_dominated(sol1, sol2):
                    is_dominated_by_others = True
                    break

            if not is_dominated_by_others:
                non_dominated_solutions.append(sol1)

        return non_dominated_solutions
    
    
    @staticmethod
    def _is_dominated(sol1, sol2):
        """
        Checks if sol1 dominated by the sol2.

        Args:
            sol1 (_type_)
            sol2 (_type_)
            
        Returns:
            bool
        """
        return sol2.val_accuracy >= sol1.val_accuracy and sol2.train_time <= sol1.train_time


def load_models(train_time_path, acc_path):
    # Load the XGBoost model from the pickle file
    with open(train_time_path, 'rb') as file:
        time_model = pickle.load(file)
        
    with open(acc_path, 'rb') as file:
        acc_model = pickle.load(file)
        
    return time_model, acc_model

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
                   pareto_steps:int,
                   starting_points:int,
                   min_max_dict: Dict,
                   result_dir: Path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis")
                   ) -> None:
    
    for size in size_models:
        surrogate_PLS(size_models=size_models,
                      dataset_api=dataset_api,
                      size=size,
                      pareto_steps=pareto_steps,
                      starting_points=starting_points,
                      min_max_dict=min_max_dict,
                      result_dir=result_dir)
        
    pass
    
    
def surrogate_PLS(size_models: Dict,
             dataset_api: Dict,
             size:int,
             pareto_steps:int,
             starting_points:int,
             min_max_dict: Dict,
             result_dir: Path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis")
             ) -> None:
    
    
    for seed in size_models[size]:
        # Get the surrogate front
        surrogate_front = sf.get_surrogate_front(dataset_api, size_models[size][seed]["val_accuracy"].model, size_models[size][seed]["train_time"].model)
        np.random.seed(seed)
        if len(surrogate_front) > starting_points:
            surrogate_front = np.random.choice(surrogate_front, size=starting_points, replace=False)
        else: 
            print(f"Surrogate front size is smaller than starting points. For the size: {size} and seed: {seed} Surrogate front size: {len(surrogate_front)}")
        
        search_res_dir = result_dir / str(size) / str(seed)
        print(f"Making surrogate MO for size {size} and seed {seed} and len(surrogate_front): {len(surrogate_front)}")
        make_PLS(search_res_dir=search_res_dir,
                 starting_archs=surrogate_front,
                 pareto_steps=pareto_steps,
                 min_max_dict=min_max_dict,
                 )
        
    pass
    

def make_PLS(search_res_dir: Path, starting_archs: List[str], max_train_cnt: int, min_max_dict: Dict) -> int:
    
    if not os.path.exists(search_res_dir):
        os.makedirs(search_res_dir)
    
    paretos = []
    #min_max_dict = metrics.get_min_max_values(dataset_api["nb101_data"])
    trained_arch_cnt = 0
    str_point_train_cnt = max_train_cnt // len(starting_archs)
    for i, arch in enumerate(starting_archs):
        starting_arch = ArchCoupled(arch, dataset_api["nb101_data"])
        PLS = ParetoLocalSearch(starting_arch, str_point_train_cnt, dataset_api)
        pareto_ls = PLS.search()
        paretos += pareto_ls
        trained_arch_cnt += PLS.trained_arch_cnt
        write_pls_history(pls=PLS, path=search_res_dir, start_point_id=i+1)
    
    full_front = PLS._find_non_dominated_solutions(paretos)
    
    pareto_metrics = metrics.ParetoMetrics(full_front, min_max=min_max_dict, trained_arch_cnt=trained_arch_cnt)
    
    write_pareto_metrics(pareto_metrics, path=search_res_dir)
    plot.plot_pareto_front(pareto_front=full_front, min_max=min_max_dict, path=search_res_dir / "pareto_front.png")
    
    return trained_arch_cnt
    
def run_surrogate_PLS(
    pareto_steps:int, 
    starting_points:int,
    min_max_dict: Dict,
    model_dir: str = "../surrogates/models",
    result_dir:Path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis/surrogates")
    ):


    size_models = get_size_models(model_dir=model_dir)
    
    multi_surrogate_PLS(size_models=size_models,
                        dataset_api=dataset_api,
                        result_dir=result_dir,
                        pareto_steps=pareto_steps,
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
    random_seeds = list(range(10,310,10))
    size_models = get_size_models(model_dir="../surrogates/models")
    raw_mo_steps = PARETO_STEPS
    raw_mo_result_path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis/raw_mo") / str(raw_mo_steps)
    surrogate_mo_result_path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis/surrogates") / str(PARETO_STEPS)
    print(raw_mo_result_path)
    
    # We make 1 to 5 steps more for raw MO.
    #for i in [0, 1, 2, 3, 4, 5]: 
    #raw_mo_steps = PARETO_STEPS + i
    raw_mo_max_train = 4000
    raw_mo_result_path = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis/raw_mo/") / str(raw_mo_max_train)
    multi_raw_MO_PLS(dataset_api=dataset_api,
                    min_max_dict=min_max_dict,
                    random_seeds=random_seeds,
                    starting_points=STARTING_POINTS,
                    max_train_cnt=raw_mo_max_train,
                    result_dir=raw_mo_result_path
                    )
    
''' 
    run_surrogate_PLS(
        pareto_steps=PARETO_STEPS,
        starting_points=STARTING_POINTS,
        min_max_dict=min_max_dict,
        result_dir=surrogate_mo_result_path,
        model_dir=Path("../surrogates/models")
        )
'''