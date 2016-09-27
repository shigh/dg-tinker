import numpy as np
from scipy.special.orthogonal import p_roots
import scipy.sparse as sps
from scipy.special import j_roots, eval_legendre

def gll_points(n):
    """GLL points and weights

    :param n: Number of points
    :returns: (x, w)
    :rtype: 

    """
    
    assert n>=2

    if n==2:
        x = np.array([-1.0, 1.0])
        w = np.array([ 1.0, 1.0])
        return x, w

    # See Nodal Discontinuous Galerkin Methods Appendix A for x and
    # the Mathworld page on Lobatto Quadrature for w
    x = j_roots(n-2, 1, 1)[0]
    L = eval_legendre(n-1, x)
    w1 = 2.0/(n*(n-1))
    w  = 2.0/(n*(n-1)*L*L)
    
    x = np.hstack([-1.0, x, 1.0])
    w = np.hstack([w1, w, w1])
    
    return x, w    

def eval_lagrange_d0(xnodes, xref):
    """Stable lagrange eval on interval ref

    :param xnodes: Lagrange nodes
    :param xref: Ref locations to eval
    :returns: phi at ref points for each basis function
    :rtype: np.ndarray: len(nxnodes) x len(xref)

    """

    nxnodes = len(xnodes)
    nx      = len(xref)

    phi1d = np.ones((nxnodes,nx),dtype=float)
    coeff = np.zeros((nxnodes,))
    for i in range(0,nxnodes):

        # compute (xi-x1)(xi-x2)...(xi-xn) for each x
        coeff[i]=1.0
        for j in range(0,nxnodes):
            if i!=j:
                coeff[i] *= xnodes[i] - xnodes[j]
                
        for k in range(0,nx):
            for j in range(0,nxnodes):
                if i!=j:
                    phi1d[i,k] *= xref[k] - xnodes[j]
            phi1d[i,k] *= 1.0/coeff[i]
            
    return phi1d

def eval_lagrange_d1(xnodes, xref):
    """Stable lagrange derivative eval on interval ref

    :param xnodes: Lagrange nodes
    :param xref: Ref locations to eval
    :returns: phi at ref points for each basis function
    :rtype: np.ndarray: len(nxnodes) x len(xref)

    """

    nxnodes = len(xnodes)
    nx      = len(xref)

    dphi1d = np.zeros((nxnodes,nx),dtype=float)
    coeff  = np.zeros((nxnodes,))
    for i in range(0,nxnodes):

        # compute (xi-x1)(xi-x2)...(xi-xn) for each x
        coeff[i]=1.0
        for j in range(0,nxnodes):
            if i!=j:
                coeff[i] *= xnodes[i] - xnodes[j]
                
        # comput d l_i at xg
        for k in range(0,nx):
            for j in range(0,nxnodes):
                if i!=j:
                    addon = 1.0
                    for l in range(0,nxnodes):
                        if (l!=j) and (l!=i):
                            addon *= xref[k]-xnodes[l]
                    dphi1d[i,k] += addon
            dphi1d[i,k] *= 1.0/coeff[i]

    return dphi1d
    

class Interval(object):
    
    interval = (-1.0, 1.0)
    h        = 2.0
    
    def calc_jacb(self, nodes):
        do_ravel = nodes.ndim==1
        if do_ravel:
            nodes = nodes.reshape((1,-1))
        
        jacb = (nodes[:,1]-nodes[:,0])/self.h
        assert np.all(jacb!=0.0)
        
        if do_ravel: return jacb.ravel()
        return jacb
    
    def calc_jacb_det(self, jacb):
        return jacb
    
    def calc_jacb_inv(self, jacb):
        return 1.0/jacb
    
    def calc_jacb_inv_det(self, jacb):
        return 1.0/jacb
    
    def get_quadrature(self, n):
        return p_roots(n)
    
    def ref_to_phys(self, nodes, ref):
        do_ravel = nodes.ndim==1
        if do_ravel:
            nodes = nodes.reshape((1,-1))
        
        a = nodes[:,0].reshape((-1,1))
        b = (nodes[:,1]-nodes[:,0])
        b = b.reshape((-1,1))
        phys = a+b*(ref+1)/self.h
        
        if do_ravel: return phys.ravel()
        return phys
    
    def phys_to_ref(self, nodes, phys):
        pass


class SEMhat(object):
    
    def __init__(self, order):

        n = order+1

        # GLL quadrature points
        xgll, wgll = gll_points(order+1)
        self.xgll, self.wgll = xgll, wgll
        
        # Mass matrix
        Bh = sps.dia_matrix((wgll, 0), shape=(n, n))
        self.Bh = Bh
        
        # Derivative evaluation matrix
        Dh = eval_lagrange_d1(xgll, xgll).T
        self.Dh = Dh
        
        # Diffusion operator
        # Ah = Dh'*Bh*Dh
        Ah = Dh.T.dot(Bh.dot(Dh))
        self.Ah = Ah
        
        # Convection operator
        Ch = Bh.dot(Dh)
        self.Ch = Ch

    def interp_mat(self, x):
        """Interpolation matrix from self.xgll to x

        :param x: Points to interpolate to
        :returns: Interpolation matrix
        :rtype: np.ndarray

        """

        J = eval_lagrange_d0(self.xgll, x)
        return J.T
