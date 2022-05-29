import numpy as np
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(dataset_name, feat_size):
        pass

def loadData(filepath, feat_size):
    """
    Load dataset from filepath and normalize the features.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
        labels = np.zeros(len(lines))
        features = np.zeros((len(lines), feat_size))
        for i, line in enumerate(lines):
            cols = line.split()
            labels[i] = int(float(cols.pop(0)))
            for col in cols:
                features[i][int(col.split(':')[0])-1] = float(col.split(':')[1])
        features = StandardScaler().fit_transform(features)
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        return features, labels

