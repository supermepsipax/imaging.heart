import nrrd

def load_nrrd_mask(path, verbose=False):
    data, header = nrrd.read(path)

    if verbose: 
        print("Shape:", data.shape)
        print("Available Header Info:")
        for heading, value in header.items():
            print(f'{heading}: {value}')

    return data
