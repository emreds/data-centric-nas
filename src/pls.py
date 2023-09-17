import copy
from collections import deque
from typing import Dict

import neighborhood


class ParetoLocalSearch:
    """
    ParetoLocalSearch performs Pareto Local Search (PLS) to find non-dominant solutions in an architectural space.

    This class initializes with a starting architecture, the number of iterations to perform, and a dataset API.

    Args:
        arch (object): The starting architecture from which PLS begins.
        iterations (int): The number of iterations to perform during PLS.
        dataset_api (dict): An API or data source for the dataset used in the search.

    Attributes:
        base_arch (object): The starting architecture.
        iterations (int): The number of iterations to perform.
        dataset_api (dict): The dataset API.
        archive (list): A list of architectures representing the Pareto front.
        iter_queue (deque): A queue for iterating through architectures.
        arch_hashes (set): A set of hash values for tracked architectures.
        trained_arch_cnt (int): The total count of trained architectures.
        history (dict): A dictionary containing the search history at each iteration.

    Methods:
        search(): Perform Pareto Local Search for the specified number of iterations.
        _find_non_dominated_solutions(pareto_front): Find non-dominant solutions in the given Pareto front.
        _is_dominated(sol1, sol2): Check if sol1 is dominated by sol2 based on validation accuracy and training time.
    """
    
    def __init__(self, arch: object, iterations: int, dataset_api: dict) -> None:
        self.base_arch = arch
        self.iterations = iterations
        self.dataset_api = dataset_api
        self.archive = [arch]
        self.iter_queue = deque(self.archive)
        self.arch_hashes = set([self.base_arch.hash])
        self.trained_arch_cnt = 0
        self.history = {0: [arch]}
    
    # (Rest of the class implementation)

    def __init__(self, arch, iterations: int, dataset_api: Dict) -> None:
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

            #print(f"This is archive length for step {i+1}: {len(self.archive)}")
        
            i += 1
            
            self.history[len(self.history)] = [copy.deepcopy(arch) for arch in self.archive]
            
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
