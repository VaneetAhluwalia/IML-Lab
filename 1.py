import numpy as np
import pandas as pd
arr = eval(input("Enter the list :"))
np_arr = np.array(arr)
pd_arr = pd.Series(arr)

print(np_arr)
print(pd_arr)
