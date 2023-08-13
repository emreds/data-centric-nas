
import numpy as np
import pandas as pd
from naslib.search_spaces.nasbench101 import encodings
from sklearn.preprocessing import MinMaxScaler

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
        if len(arch["module_operations"]) == 7:
            encoding = _get_encoding(arch)
            encodings[arch_hash] = encoding
    return encodings

def get_surrogate_front(dataset_api: dict, acc_model, time_model):

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

    scaler = MinMaxScaler()
    df["norm_val_acc"] = scaler.fit_transform(df["val_acc"].values.reshape(-1, 1)).round(2)
    df["norm_train_time"] = scaler.fit_transform(df["train_time"].values.reshape(-1, 1)).round(2)
    df["score"] = (df["norm_val_acc"] + ((1 - df["train_time"]))/2)/2

    pareto_front = df.sort_values(by="score", ascending=False).head(20)["arch_hash"].tolist()
    
    return pareto_front
