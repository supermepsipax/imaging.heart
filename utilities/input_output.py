import numpy as np
import nrrd

def load_nrrd_mask(path):
    data, header = nrrd.read(path)
    print("Shape:", data.shape)
    print("Spacing:", header.get("space directions"))
    return data
