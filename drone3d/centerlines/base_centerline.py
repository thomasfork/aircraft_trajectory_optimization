''' base centerline classes '''
from dataclasses import dataclass, field
from abc import abstractmethod, ABC
from typing import List, Callable, Dict
from functools import singledispatch
from enum import Enum

import numpy as np
import casadi as ca

from scipy.spatial.distance import cdist

from drone3d.visualization.shaders import vtype
from drone3d.pytypes import PythonMsg, \
    RacerState, DroneState, RelativeOrientation
from drone3d.visualization.objects import VertexObject, InstancedVertexObject, UBOObject
from drone3d.visualization.utils import get_circular_gate, get_unit_arrow, join_vis, \
    get_square_gate, get_instance_transforms

class GateShape(Enum):
    ''' gate shape options '''
    CIRCLE = 0
    SQUARE = 1

@dataclass
class BaseCenterlineConfig(PythonMsg):
    ''' standard or necessary parameters for a centerline surface '''
    #TODO manually specifying gate pose as an option besides auto-generation
    #TODO enable different shapes for different gates
    # global bounds on all variables (ignoring regularity limits)
    s_min: float = field(default = 0)
    s_max: float = field(default = 10)
    y_min: float = field(default = -2)
    y_max: float = field(default = 2)
    n_min: float = field(default = -2)
    n_max: float = field(default = 2)

    # gate parameters
    gate_s: np.ndarray = field(default = None)
    gate_shape: GateShape = field(default = GateShape.CIRCLE)
    gate_ri: float = field(default = 1.25)
    gate_ro: float = field(default = 1.35)
    gate_w:  float = field(default = 0.2)
    gate_snap_fit: bool = field(default = True)

    # whether or not it is periodic
    closed:bool = field(default = False)

    # gamma parameter for regularity
    gamma: float = field(default = 0.9)

    # number of grid points on centerline for global to local conversion
    # larger numbers are more accurate but slow down global to local conversion
    N_grid: int = field(default = 10000)


@dataclass
class BaseSymRep(PythonMsg):
    ''' symbolic terms for any centerline '''
    x: ca.SX = field(default = None)
    xc: ca.SX = field(default = None)
    p: ca.SX = field(default = None)
    es: ca.SX = field(default = None)
    ey: ca.SX = field(default = None)
    en: ca.SX = field(default = None)
    Rp: ca.SX = field(default = None)
    ks: ca.SX = field(default = None)
    ky: ca.SX = field(default = None)
    kn: ca.SX = field(default = None)
    k:  ca.SX = field(default = None)
    mag_xcs: ca.SX = field(default = None)
    param_terms: ca.SX = field(default = None)
    f_param_terms: ca.Function = field(default = None)


class BaseCenterline(ABC):
    ''' base centerline class, to describe shape and geometry of local coordinates '''
    config: BaseCenterlineConfig
    sym_rep: BaseSymRep

    # cleanly closed, ie. no discontinuity in y,n variables at start/end
    cleanly_closed = True

    ## functions for various useful terms, returned as numpy arrays or floats
    ## not for symbolic evaluation.
    # global position from (s,y,n)
    p2x: Callable[[float, float, float], np.ndarray]
    p2xc: Callable[[float], np.ndarray]

    # darboux frame basis
    p2es:   Callable[[float], np.ndarray]
    p2ey:   Callable[[float], np.ndarray]
    p2en:   Callable[[float], np.ndarray]
    # orientation matrix of parametric frame
    p2Rp: Callable[[float], np.ndarray]

    # darboux frame curvatures from (s)
    p2k:  Callable[[float], np.ndarray]
    p2ks: Callable[[float], float]
    p2ky: Callable[[float], float]
    p2kn: Callable[[float], float]

    # grid of points along xc with rows [s, xi, xj, xk] for global to local conversion
    xc_grid: np.ndarray

    # attributes for visualizing the centerline nicely: a central point and a scale
    # populated when the centerline is triangulated
    view_center: np.ndarray
    view_scale: float

    def __init__(self, config: BaseCenterlineConfig):
        self.config = config
        self._setup_interp()
        self._compute_sym_rep()
        self._unpack_sym_rep()

    def fill_in_centerline(self,
            name: str,
            expr: ca.SX,
            args:List[ca.SX],
            p = None) -> ca.Function:
        '''
        takes an expr that is a function of args and param_terms
        and makes param_terms implicit from self.f_param_terms
        '''
        if not isinstance(expr, list):
            expr = [expr]
        else:
            expr = [ca.vertcat(*expr)]

        if not isinstance(args, list):
            args = [args]

        if p is None:
            p = self.sym_rep.p

        s = p[0]

        func = ca.Function(name, [*args, self.sym_rep.param_terms], expr)
        feval = func(*args, self.sym_rep.f_param_terms(s))
        filled_func = ca.Function(name, args, [feval])

        def f(*args):
            return np.array(filled_func(*args)).squeeze()

        def f_ca(*args):
            return filled_func(*args)

        f = singledispatch(f)
        f.register(ca.SX, f_ca)
        f.register(ca.MX, f_ca)
        f.register(ca.DM, f_ca)

        return f

    def s_min(self):
        '''
        minimum s coordinate of the centerline
        '''
        return self.config.s_min

    def s_max(self):
        '''
        maximum s coordinate of the centerline
        '''
        return self.config.s_max

    def y_min(self, s: float = 0):
        '''
        minimum y coordinate of the centerline
        '''
        return self.config.y_min

    def y_max(self, s: float = 0):
        '''
        maximum y coordinate of the centerline
        '''
        return self.config.y_max

    def n_min(self, s: float = 0):
        '''
        minimum n coordinate of the centerline
        '''
        return self.config.n_min

    def n_max(self, s: float = 0):
        '''
        maximum n coordinate of the centerline
        '''
        return self.config.n_max

    def fast_p2x(self, s, y, n) -> np.ndarray:
        '''
        helper function for texture generation that takes vector variables
        and returns 3D position vectors for each

        redundant for some surfaces; for others it supports vectorized p for faster,
        numeric-only evaluation.
        '''
        return self.p2x(s, y, n)

    def fast_p2ey(self, s) -> np.ndarray:
        '''
        helper function for texture generation that takes a vectorized pose, ie. 3xN array
        and returns normal vectors for each
        '''
        return self.p2ey(s)

    def fast_p2en(self, s) -> np.ndarray:
        '''
        helper function for texture generation that takes a vectorized pose, ie. 3xN array
        and returns normal vectors for each
        '''
        return self.p2en(s)

    def l2gx(self, state:RacerState):
        '''
        parametric position -> global position
        '''
        state.x.from_vec(self.p2x(*state.p.to_vec()))

    def l2gq(self, state:DroneState):
        ''' converts state.p and state.r pose variables into state.q '''
        if isinstance(state.r, RelativeOrientation):
            Rp = self.p2Rp(state.p.to_vec())
            R = Rp @ state.r.R()
        else:
            R = state.r.R()
        state.q.from_mat(R)

    def g2lx(self, state:RacerState):
        '''
        global position -> parametric position
        '''
        x_query = state.x.to_vec()[np.newaxis, :]

        s = self.xc_grid[cdist(x_query, self.xc_grid[:,1:], metric = 'sqeuclidean').argmin(), 0]
        xc = self.p2xc(s)
        es = self.p2es(s)
        ey = self.p2ey(s)
        en = self.p2en(s)
        s += float((x_query - xc) @ es)
        state.p.s = s
        state.p.y = float((x_query - xc) @ ey)
        state.p.n = float((x_query - xc) @ en)

    def x2p(self, x_query: np.ndarray):
        '''
        convert global position(s)
        to parametric position(s)
        x should be a Nx3 array
        '''
        s: np.ndarray = self.xc_grid[
            cdist(x_query, self.xc_grid[:,1:], metric = 'sqeuclidean').argmin(axis = 1),
            0]
        Rp = self.p2Rp(s[None]).reshape((3,-1, 3)).transpose((1,0,2))
        xc = self.p2xc(s[None]).T
        p = np.zeros((s.shape[0], 3))
        for k in range(s.shape[0]):
            #TODO - could vectorize this
            p[k, 0] = s[k] + (x_query[k] - xc[k]) @ Rp[k,:,0]
            p[k, 1] = (x_query[k] - xc[k]) @ Rp[k,:,1]
            p[k, 2] = (x_query[k] - xc[k]) @ Rp[k,:,2]
        return p

    @abstractmethod
    def _setup_interp(self):
        ''' set up interpolation for the centerline '''

    @abstractmethod
    def _compute_sym_rep(self):
        ''' compute symbolic representation of parameterization'''

    def _unpack_sym_rep(self):
        '''
        unpack the major parts of the surface representation and
        create functions for them for evaluation,
        ie. converting parametric state to global state
        '''
        p = self.sym_rep.p
        s = p[0]

        x = self.sym_rep.x
        xc = self.sym_rep.xc
        mag_xcs = self.sym_rep.mag_xcs

        es = self.sym_rep.es
        ey = self.sym_rep.ey
        en = self.sym_rep.en
        Rp = ca.horzcat(es, ey, en)
        self.sym_rep.Rp = Rp

        ks = self.sym_rep.ks
        ky = self.sym_rep.ky
        kn = self.sym_rep.kn
        k = ca.vertcat(ks, ky, kn)
        self.sym_rep.k = k

        # create functions to evaluate the above for current surface
        self.p2x  = self.fill_in_centerline('x',  x,  [p[0], p[1], p[2]])
        self.p2xc = self.fill_in_centerline('xc', xc, s)
        self.p2mag_xcs = self.fill_in_centerline('mag_xcs', mag_xcs, s)

        self.p2es = self.fill_in_centerline('es', es, s)
        self.p2ey = self.fill_in_centerline('ey', ey, s)
        self.p2en = self.fill_in_centerline('en', en, s)
        self.p2Rp = self.fill_in_centerline('Rp', Rp, s)

        self.p2ks = self.fill_in_centerline('ks', ks, s)
        self.p2ky = self.fill_in_centerline('ky', ky, s)
        self.p2kn = self.fill_in_centerline('kn', kn, s)
        self.p2k  = self.fill_in_centerline('k',  k,  s)

    def gate_position(self, s: float) -> np.ndarray:
        ''' return vector for gate position at s '''
        return self.p2xc(s)

    def gate_orientation(self, s: float) -> np.ndarray:
        ''' return orientation matrix for gate position at s'''
        es = self.p2es(s)
        ey = self.p2ey(s)
        en = self.p2en(s)


        if self.config.gate_snap_fit:
            if abs(es[2]) > 0.9:
                es = np.array([0,0,1])
                ey = ey - es * (ey.T @ es)
                ey = ey / np.linalg.norm(ey)
                en = np.cross(es, ey)
            elif abs(en[2]) > 0.9:
                en = np.array([0,0,1])
                es = es - en * (es.T @ en)
                es = es / np.linalg.norm(es)
                ey = np.cross(en, es)
        return np.array([es, ey, en]).T

    def _triangulate_domain(
            self,
            n: int = 1000,
            m = 50):

        '''
        generate vertex/index data for regular domain of local coords with factor gamma <=1
        '''
        s = np.linspace(self.s_min(), self.s_max()-0.001, n)
        x = self.fast_p2x(s, 0, 0).T
        ey = self.fast_p2ey(s).T
        en = self.fast_p2en(s).T

        ky = -1 * self.p2ky(s)
        kn = self.p2kn(s)

        ky_full = np.kron(ky, np.ones(m))
        kn_full = np.kron(kn, np.ones(m))
        k_sq_full = np.kron(ky**2 + kn**2, np.ones(m))

        th = np.linspace(0, 2*np.pi, m, endpoint=True)

        alpha_y = (np.cos(th) > 0) * self.config.y_max + (np.cos(th) <=0) * self.config.y_min
        alpha_n = (np.sin(th) > 0) * self.config.n_max + (np.sin(th) <=0) * self.config.n_min
        #alpha_y = np.cos(th) * self.config.y_max
        #alpha_n = np.sin(th) * self.config.n_max

        apo = np.kron(ky, alpha_n) + np.kron(kn, alpha_y)
        out_of_bounds = np.argwhere(apo > self.config.gamma)

        alpha_y_full = np.kron(np.ones(n), alpha_y)
        alpha_n_full = np.kron(np.ones(n), alpha_n)
        alpha_y_full[out_of_bounds] = alpha_y_full[out_of_bounds] \
            - kn_full[out_of_bounds] * (apo[out_of_bounds] - self.config.gamma) \
                / k_sq_full[out_of_bounds]
        alpha_n_full[out_of_bounds] = alpha_n_full[out_of_bounds] \
                - ky_full[out_of_bounds] * (apo[out_of_bounds] - self.config.gamma) \
                / k_sq_full[out_of_bounds]

        x  = np.kron(x,  np.ones((m,1)))
        delta_y = np.kron(ey, np.ones((m,1))) * alpha_y_full[:,np.newaxis]
        delta_n = np.kron(en, np.ones((m,1))) * alpha_n_full[:,np.newaxis]

        N = np.kron(ey, np.cos(th)[:,np.newaxis]) + np.kron(en, np.sin(th)[:,np.newaxis])
        I = np.zeros((n-1, m-1, 2,3))

        idxs = np.arange(n*m).reshape(n,m)

        I[:,:,0,0] = idxs[:-1,:-1]       # bot left
        I[:,:,0,2] = idxs[1: ,:-1]       # bot right
        I[:,:,0,1] = idxs[:-1,1: ]       # top left
        I[:,:,1,0] = idxs[1: ,1: ]       # top right
        I[:,:,1,2] = idxs[:-1,1: ]       # top left
        I[:,:,1,1] = idxs[1: ,:-1]       # bot right

        I = I.reshape(-1,3)

        if not self.config.closed:
            I_start = []
            I_end = []
            ifo = m*(n-1)
            for k in range(m-1):
                I_start.append([k+1, k, m-1])
                I_end.append([ifo+k, ifo+k+1, ifo+m-1])
            I = np.concatenate([I, np.array(I_start), np.array(I_end)])

        Vertices = np.zeros(x.shape[0], dtype=vtype)
        Vertices['a_position'] = x + delta_y + delta_n
        Vertices['a_color']    = np.array([0.6,0.6,0.6, 0.2])
        Vertices['a_normal']   = N

        return Vertices, np.concatenate(I).astype(np.uint32)

    def _triangulate_centerline(
            self,
            n: int = 1000,
            w: float = 0.05,
            h: float = 0.05):
        '''
        turn the centerline
        into vertex/index data for plotting
        '''
        # generate data
        s = np.linspace(self.s_min(), self.s_max(), n)
        x = self.fast_p2x(s, 0, 0).T
        e2 = self.fast_p2ey(s).T
        e3 = self.fast_p2en(s).T

        V = np.concatenate([x + e2*w + e3*h,
                            x + e2*w,
                            x - e2*w,
                            x - e2*w + e3*h])

        x_max = V.max(axis = 0)
        x_min = V.min(axis = 0)
        self.view_center = (x_max + x_min) / 2
        self.view_scale   = np.linalg.norm(x_max - x_min)

        N = np.zeros(V.shape)
        N[:,-1] = 1

        # first face
        I = np.array([[0,1,n],
                    [1,n+1,n]])
        # first segment
        I = np.concatenate([I, I+n, I+2*n, I+3*n])
        I = I % (n*4)
        #all segments
        I = np.concatenate([I + k for k in range(n-1)])

        if not self.config.closed:
            # cap the ends
            I = np.concatenate([I, np.array([[0,n,2*n],
                                            [0,2*n, 3*n]])])
            I = np.concatenate([I, n-1+ np.array([[0,2*n,n],
                                        [0,3*n, 2*n]])])
        else:
            # join start and end
            I_join =  np.array([[0,n,n-1], [n,2*n-1,n-1]])
            I_join = np.concatenate([I_join, I_join+n, I_join+2*n, I_join+3*n]) % (4*n)

            I = np.concatenate([I, I_join])

        Vertices = np.zeros(V.shape[0], dtype=vtype)
        Vertices['a_position'] = V
        Vertices['a_color']    = np.array([0.2,0.2,0.2, 1])
        Vertices['a_normal']   = N

        return Vertices, np.concatenate(I).astype(np.uint32)

    def generate_texture(self, ubo: UBOObject) -> Dict[str, VertexObject]:
        ''' generate various objects to plot in 3D'''
        objects = {}

        V, I = self._triangulate_centerline()
        objects['Centerline'] = VertexObject(ubo, V, I)

        if self.config.gate_s is not None:
            V, I = generate_gate_textures(self, self.config.gate_s)
            objects['Gates'] = VertexObject(ubo, V, I)

        V, I = self._triangulate_domain()
        objects['Local Domain'] = VertexObject(ubo, V, I)

        coords = InstancedVertexObject(ubo, *join_vis(
            [
                get_unit_arrow(d=1, color = [1,0,0,1]),
                get_unit_arrow(d=2, color = [0,1,0,1]),
                get_unit_arrow(d=3, color = [0,0,1,1])
            ]
        ))

        if self.config.gate_s is not None:
            s = np.linspace(self.s_min(), self.s_max(), len(self.config.gate_s)*5)[None]
        else:
            s = np.linspace(self.s_min(), self.s_max(), 100)[None]

        x = self.p2xc(s).T
        r = self.p2Rp(s)
        r = r.T.reshape((-1,3,3))
        r = r.transpose((0,2,1))
        R = get_instance_transforms(
            x = x,
            r = r
        )
        coords.apply_instancing(R)
        objects['Local Coords'] = coords

        return objects

    def preview_centerline(self):
        '''
        plot the centerline all by itself
        '''
        # pylint: disable=import-outside-toplevel
        # disabled to avoid circular import and since speed is not critical here
        from drone3d.visualization.opengl_fig import Window
        window = Window(self)

        should_close = False
        while not should_close:
            should_close = window.draw()
        window.close()


def generate_gate_textures(
        line: BaseCenterline,
        s: List[float],
        n: int = 500):
    ''' generate vertex data for visualizing a sequence of gates to go through '''
    Vertices = None
    Indices = None
    ri = line.config.gate_ri
    ro = line.config.gate_ro
    w = line.config.gate_w
    for sk in s:
        if line.config.gate_shape == GateShape.CIRCLE:
            V, I = get_circular_gate(n, ri, ro, w)
        elif line.config.gate_shape == GateShape.SQUARE:
            V, I = get_square_gate(ri, ro, w)
        else:
            raise NotImplementedError('unhandled gate shape')

        x0 = line.gate_position(sk)
        R  = line.gate_orientation(sk).T

        V['a_position'] = V['a_position'] @ R + x0
        V['a_normal'] = V['a_normal'] @ R

        if Vertices is None or Indices is None:
            Vertices = V
            Indices = I
        else:
            Indices = np.concatenate([Indices, I + len(Vertices)])
            Vertices = np.concatenate([Vertices, V])

    return Vertices, Indices.astype(np.uint32)
