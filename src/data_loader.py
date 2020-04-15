import numpy as np
import h5py

def load_segmentation(input_filepath, column, idx):
    with h5py.File(input_filepath, 'r') as hf:
        m = np.asarray(hf.get(column)[idx])
    return m

def load_vector_fields(input_filepath, columns, idx):
    with h5py.File(input_filepath, 'r') as hf:
        u = np.asarray(hf.get(columns[0])[idx])
        v = np.asarray(hf.get(columns[1])[idx])
        w = np.asarray(hf.get(columns[2])[idx])
    return u, v, w