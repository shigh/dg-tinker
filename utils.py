
import numpy as np

def minmod(a, b, c):

    A = np.dstack([a,b,c])
    # Drop the dimension added by dstack
    A = A.reshape((-1,3))

    s = np.sum(np.sign(A),
               axis=1)/3.0
    ret = np.zeros(len(s), 
                   dtype=np.double)
    ind = np.abs(s)==1.0

    ret[ind] = s[ind]*np.min(np.abs(A[ind]),
                             axis=1)
    return ret
