
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
        self.periodic = periodic

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

class LimiterMUSCL(object):

    def __init__(self, mesh):

        self.mesh = mesh
        N = mesh.N

        # elem midpoints
        x0 = (mesh.xq[:,0]+mesh.xq[:,-1])/2.0
        # elem widths (assuming uniform)
        h  = (mesh.xq[:,-1]-mesh.xq[:,0])[0]
        self.x0, self.h = x0, h

        # Remove scale factors built into V
        # (b.c. of the def used in nodal-dg)
        nv  = np.arange(N+1)
        gam = 2.0/(2.0*nv+1)
        G = sps.dia_matrix((1.0/np.sqrt(gam),0),
                          shape=mesh.Vinv.shape)
        G = G.dot(mesh.Vinv)
        self.G = G

    def slope_limit(self, U):

        x0, h, G = self.x0, self.h, self.G
        mesh = self.mesh
        N = mesh.N
        periodic = mesh.periodic

        if not periodic:
            u = U
        else:
            u = np.zeros(U.shape[0]+2*(N+1))
            u[(N+1):-(N+1)] = U

        us = u.reshape((-1,N+1)).T
        if periodic:
            us[:,0] = us[:,-2]
            us[:,-1] = us[:,1]

        avg, slope = G.dot(us)[[0,1]]

        # The two comes from the domain of Legendre polys
        slope *= 2.0/h
        u = u.reshape((-1,N+1))

        h2 = h/2.0
        h2 = h
        m = minmod(slope[1:-1],
                   (avg[2:]-avg[1:-1])/h2,
                   (avg[1:-1]-avg[:-2])/h2)

        # xq has shape (n_elem, n_dof_per_elem)
        # This is why the rest of the arrays need to use newaxis
        xq = mesh.xq
        if periodic:
            u[1:-1] = avg[1:-1,newaxis]+(xq-x0[:,newaxis])*m[:,newaxis]
        else:
            u[1:-1] = avg[1:-1,newaxis]+(xq-x0[:,newaxis])[1:-1]*m[:,newaxis]

        if periodic:
            U[:] = u[1:-1].reshape(U.shape)

    def apply_limiter(self, u):
        self.slope_limit(u[:,0])
        self.slope_limit(u[:,1])
        self.slope_limit(u[:,2])

class LimiterNULL(object):

    def __init__(self, mesh):
        pass

    def apply_limiter(self, u):
        pass


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
