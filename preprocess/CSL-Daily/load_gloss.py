import numpy as np


path = r'/data/che_xiao/my_project/AdaptSign-main/preprocess/CSL-Daily/gloss_dict.npy'
data = np.load(path, allow_pickle=True).item()
print(len(data))