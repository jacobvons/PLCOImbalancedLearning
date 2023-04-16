import numpy as np
import pandas as pd


def read_data(path):
    return pd.read_csv(path)


def binarise(pred_proba, threshold):
    result = []
    for proba in pred_proba:
        neg_class_proba = proba[0]
        if neg_class_proba > threshold:
            result.append(0)
        else:
            result.append(1)
    return np.array(result)
