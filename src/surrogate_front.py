
from typing import List

import numpy as np
import pandas as pd
from model import Model
from naslib.search_spaces.nasbench101 import encodings


def _get_encoding(arch):
    arch["matrix"] = arch["module_adjacency"]
    arch["ops"] = arch["module_operations"]
    
    #print(arch)
    encoding = encodings.encode_adj(arch)
    pred_format = np.array(encoding)
    
    return pred_format

def get_encodings(archs):
    encodings = {}
    for arch_hash, arch in archs.items():
        #print(arch)
        # Some architectures are wrong. They have less than 7 operations but 7 adjacency matrix rows.
        if len(arch["module_operations"]) == 7:
            encoding = _get_encoding(arch)
            encodings[arch_hash] = encoding
    return encodings



def get_surrogate_front(dataset_api: dict, acc_model: Model, time_model: Model) -> List[str]:
    
    archs = dataset_api["nb101_data"].fixed_statistics

    arch_encodings = get_encodings(archs)
    encoded_archs = np.array(list(arch_encodings.values()))

    acc_predictions = acc_model.predict(encoded_archs)
    time_predictions = time_model.predict(encoded_archs)

    arch_hashes = np.array(list(arch_encodings.keys()))

    data = {
        'arch_hash': arch_hashes,
        'val_acc': acc_predictions,
        'train_time': time_predictions
    }

    df = pd.DataFrame(data)

    # Convert relevant columns to NumPy arrays
    val_accuracy = df['val_acc'].values
    train_time = df['train_time'].values

    # Sort indices based on val_accuracy
    sorted_indices = np.argsort(val_accuracy)[::-1]

    # Initialize the Pareto front with the first architecture
    pareto_indices = [sorted_indices[0]]

    # Find architectures that dominate
    for idx in sorted_indices[1:]:
        if np.all(train_time[idx] <= train_time[pareto_indices]):
            pareto_indices.append(idx)

    # Create a new DataFrame for the Pareto front
    pareto_front_df = df.iloc[pareto_indices]
    
    #print(f"Second Front DF: {pareto_front_df}")
    
    pareto_front_hashes = pareto_front_df['arch_hash'].tolist()
    
    return pareto_front_hashes
