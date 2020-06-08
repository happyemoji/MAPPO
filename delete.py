import numpy as np
import pdb
a = [0,1,3,5,7,9,10]
b = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
del_num = 0
for i in range(len(a)):
    # pdb.set_trace()
    del b[a[i]-del_num]
    del_num += 1
pdb.set_trace()
