
import numpy as np
from numpy import newaxis
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from pyfem.topo import Interval
from pyfem.poly import gll_points
from pyfem.sem import SEMhat
from pyfem.poly import eval_lagrange_d0 as eval_phi1d

from poly import eval_P
from utils import minmod

class Mesh(object):

    def __init__(self, K, N, L=1.0, periodic=False):

        self.K, self.N, self.L = K, N, L

        semh = SEMhat(N)
        n_dofs = K*(N+1)

        vertices  = np.linspace(0, L, K+1)
        EtoV      = np.zeros((K, 2), dtype=np.int)
        EtoV[:,0] = np.arange(K)
        EtoV[:,1] = np.arange(K)+1
        self.vertices = vertices
        self.EtoV = EtoV

        topo  = Interval()
        xq = topo.ref_to_phys(vertices[EtoV], semh.xgll)
        jacb_det = topo.calc_jacb(vertices[EtoV])[0]
        dx = np.min(xq[0,1:]-xq[0,:-1])
        self.dx, self.xq = dx, xq
        if periodic:
            EtoV[-1,-1] = EtoV[0,0]

        # Restriction operator
        # Removes contribution to dirch points from flux operators
        R = sps.eye(n_dofs).tocsr()
        if not periodic:
            R[0,0] = R[-1,-1] = 0.0
        self.R = R

        # Make elem to dof map
        EtoD = np.arange(K*(N+1))
        EtoD = EtoD.reshape((K, -1))
        self.EtoD = EtoD
        dof_phys = xq.ravel()
        self.dof_phys = dof_phys

        # Averaging operator
        rows = EtoD[:,[0,-1]].ravel()
        cols = EtoV.ravel()
        vals = np.ones_like(cols)

        FtoD = sps.coo_matrix((vals, (rows, cols))).tocsr()
        AVG = FtoD.dot(FtoD.T)/2.0
        self.AVG = AVG

        # Extract face DOFS
        vals = np.ones(len(rows))
        FD = sps.coo_matrix((vals, (rows, rows))).tocsr()
        # Set face signs
        vals[::2] = -1
        SD = sps.coo_matrix((vals, (rows, rows))).tocsr()
        self.FD, self.SD = FD, SD

        # Jump operator
        JUMP = FtoD.dot(SD.dot(FtoD).T)
        self.JUMP = JUMP

        # Build Advection operator
        C = sps.kron(sps.eye(K), semh.Ch).tocsr()
        self.C = C

        # Build full elemental mass matrix
        x, w = topo.get_quadrature(N+1)
        P = eval_phi1d(semh.xgll, x).T
        G = sps.dia_matrix((w, 0), shape=(len(x), len(x)))
        Bf = P.T.dot(G.dot(P))*jacb_det
        Bfinv = np.linalg.inv(Bf)

        # Using trick from book
        V    = eval_P(N, semh.xgll).T
        Vinv = np.linalg.inv(V)
        Minv = V.dot(V.T)/jacb_det
        Binv = sps.kron(sps.eye(K), Minv).tocsr()
        self.Binv = Binv
        self.Vinv, self.V = Vinv, V

class EqnSetEuler(object):

    def __init__(self, gamma=7.0/5.0):
        self.gamma = gamma

    def calc_flux(self, u):

        gamma = self.gamma
        f = np.zeros_like(u)
        p = (gamma-1)*(u[:,2]-.5*u[:,1]**2/u[:,0])
        f[:,0] = u[:,1]
        f[:,1] = u[:,1]**2/u[:,0]+p
        f[:,2] = (u[:,2]+p)*u[:,1]/u[:,0]

        return f

    def calc_eig(self, u):
        p = self.calc_p(u)
        gamma = self.gamma
        eig  = np.abs(u[:,1]/u[:,0])
        eig += np.sqrt(gamma*p/u[:,0])
        return eig

    def calc_p(self, u):
        gamma = self.gamma
        p = (gamma-1)*(u[:,2]-.5*u[:,1]**2/u[:,0])
        return p
