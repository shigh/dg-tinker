
import numpy as np

def eval_P(order, ref):
    """Evaluate scaled Legendre polynomials

    Using the definition from nodal-dg chapter 3. Returns a matrix
    equivalent to V^T from the same chapter.

    :param order: Highest polynomial order
    :param ref: (n_ref) Ref points ([-1,1]) to evaluate at
    :returns: (order+1, n_ref) Polynomials evaluated at points
    :rtype: np.ndarray

    """
    
    assert order>=1
    assert ref.ndim==1
    
    res = np.zeros((order+1, len(ref)))

    res[0,:] = 1.0/np.sqrt(2.0)
    res[1,:] = ref*np.sqrt(3.0/2.0)
    
    nv = np.arange(order+1)
    a  = np.sqrt(nv*nv/((2.*nv+1.)*(2.*nv-1.)))
    for n in range(2, order+1):
        res[n,:] = (ref*res[n-1,:]-a[n-1]*res[n-2,:])/a[n]
        
    return res
