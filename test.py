import numpy as np

a = np.load("/data/HumanML3D/dataset/old_HumanML3D/Std.npy")
b = np.load("/data/dataset/HumanML3D/Std.npy")

print(a-b)