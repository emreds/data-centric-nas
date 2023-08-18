import copy
import json
import os
import pickle
from collections import defaultdict, deque
from pathlib import Path

import model
import neighborhood
import plot
import surrogate_front as sf
from metrics import ParetoMetrics
from naslib.utils import get_dataset_api
from schema import ArchCoupled


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


def load_models(train_time_path, acc_path):
    # Load the XGBoost model from the pickle file
    with open(train_time_path, 'rb') as file:
        time_model = pickle.load(file)
        
    with open(acc_path, 'rb') as file:
        acc_model = pickle.load(file)
        
    return time_model, acc_model

def write_pareto_metrics(pareto_metrics: ParetoMetrics, path: Path) -> None:
    
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
# Check out the inside of full_front
def make_pls(size_models, result_dir = Path("../analysis")) -> None:
    
        for size in size_models:
            if not os.path.exists(result_dir / size):
                os.mkdir(result_dir / size)
            total_trained_arch = 0
            for seed in size_models[size]:
                # Get the surrogate front
                surrogate_front
                surrogate_front = sf.get_surrogate_front(dataset_api, size_models[size]["val_accuracy"].model, size_models[size]["train_time"].model)
                search_res_dir = result_dir / size / seed
                if not os.path.exists(search_res_dir):
                    os.mkdir(search_res_dir)
                paretos = []
                
                for i, arch in enumerate(surrogate_front):
                    starting_arch = ArchCoupled(arch, dataset_api["nb101_data"])
                    PLS = ParetoLocalSearch(starting_arch, 100, dataset_api)
                    pareto_ls = PLS.search()
                    paretos += pareto_ls
                    total_trained_arch += PLS.trained_arch_cnt
                    write_pls_history(pls=PLS, path=search_res_dir, start_point_id=i+1)
                
                full_front = PLS._find_non_dominated_solutions(paretos)
                #pareto_metrics = ParetoMetrics(full_front)
                
                #write_pareto_metrics(pareto_metrics, path=search_res_dir)
                plot.plot_pareto_front(full_front, path = search_res_dir / "pareto_front_pls.png")
        pass
    
def demo_make_pls(size_models, result_dir = Path("/p/project/hai_nasb_eo/emre/data_centric/data-centric-nas/analysis"), size=300, pareto_steps=20):
    
    #size_models = size_models[300]
    #size = 300
    total_trained_arch = 0
    for seed in size_models[size]:
        # Get the surrogate front
        surrogate_front = sf.get_surrogate_front(dataset_api, size_models[size][seed]["val_accuracy"].model, size_models[size][seed]["train_time"].model)
        search_res_dir = result_dir / str(size) / str(seed)
        if not os.path.exists(search_res_dir):
            os.makedirs(search_res_dir)
        paretos = []
        
        for i, arch in enumerate(surrogate_front):
            starting_arch = ArchCoupled(arch, dataset_api["nb101_data"])
            PLS = ParetoLocalSearch(starting_arch, pareto_steps, dataset_api)
            pareto_ls = PLS.search()
            paretos += pareto_ls
            total_trained_arch += PLS.trained_arch_cnt
            write_pls_history(pls=PLS, path=search_res_dir, start_point_id=i+1)
        
        full_front = PLS._find_non_dominated_solutions(paretos)
        #pareto_metrics = ParetoMetrics(full_front)
        
        #write_pareto_metrics(pareto_metrics, path=search_res_dir)
        plot.plot_pareto_front(full_front, path = search_res_dir / "pareto_front.png")
        
    pass
    

if __name__ == "__main__":
    dataset_api = get_dataset_api("nasbench101", "cifar10")
    # Model Name
    # /xgb_model_cifar10_300_seed_17_val_accuracy.pkl
    
    model_paths = os.listdir("../surrogates/models")
    model_paths.sort()
    model_paths = [os.path.join("../surrogates/models", file) for file in model_paths]
    
    surr_models = [model.Model(model_path) for model_path in model_paths]

    size_models = defaultdict(lambda: defaultdict(dict))
    for model_path in model_paths:
        surr_model = model.Model(model_path)
        size_models[surr_model.data_size][surr_model.seed][surr_model.metric] = surr_model
        
    demo_make_pls(size_models=size_models, size=1000)
        
        
    #print(size_models)
    # 300,400,500
    

    
# First get the non dominated neighbours + current arch.
# Then, compare them with the archive.
# Remove the dominated ones from the archive.
# Create folder name with data size, and save the plot there. 
# Add plot of pareto front metrics. 
# Parse the archive. 
# For the archive, check the average acc and training time and plot it with every step. 
# Add the non dominated ones to the archive and iter_queue.
