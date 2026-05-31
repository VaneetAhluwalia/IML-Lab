import numpy as np
import pandas as pd

arr = eval(input("Enter the 2D list :"))

np_arr = np.array(arr).flatten()
pd_arr = pd.DataFrame(arr).values.flatten()

print(np_arr)
print(pd_arr)