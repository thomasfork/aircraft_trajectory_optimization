''' base dynamics features '''
from typing import List, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

import casadi as ca
from drone3d.pytypes import RacerState, RacerConfig, PythonMsg
from drone3d.utils.integrators import idas_integrator
from drone3d.centerlines.base_centerline import BaseCenterline

@dataclass
class DynamicsModelVars(PythonMsg):
    ''' structure for symbolic expressions for building up dynamic models '''
    # position and derivative (parametric or global depending on model)
    p: ca.SX = field(default = None)
    p_dot: ca.SX = field(default = None)
    # orientation variables (if any) and derivative
    r: ca.SX = field(default = None)
    r_dot: ca.SX = field(default = None)
    # velocity in body and global frames, and body frame derivative
    vb: ca.SX = field(default = None)
    vg: ca.SX = field(default = None)
    vb_dot: ca.SX = field(default = None)
    # angular velocity in body frame and derivative (if part of model)
    wb: ca.SX = field(default = None)
    wb_dot: ca.SX = field(default = None)
    # angular velocity of darboux frame
    wp: ca.SX = field(default = None)
    # effective angular velocity on relative orientation
    w_eff: ca.SX = field(default = None)
    # complete state vector
    z: ca.SX = field(default = None)
    # complete input vector
    u: ca.SX = field(default = None)
    # state derivative
    z_dot: ca.SX = field(default = None)
    # orientation of body relative to global frame
    R: ca.SX = field(default = None)
    # orientation of body relative to parametric frame
    R_rel: ca.SX = field(default = None)
    # net body force
    Fb: ca.SX = field(default = None)
    # net body torque
    Kb: ca.SX = field(default = None)
    # body frame gravity forces
    Fgb: ca.SX = field(default = None)
    # body, global and parametric frame thrust
    Tb: ca.SX = field(default = None)
    Tg: ca.SX = field(default = None)
    Tp: ca.SX = field(default = None)


class DynamicsModel(ABC):
    ''' base dynamics model'''
    config: RacerConfig
    _model_vars: DynamicsModelVars

    # basic expected functions for simulating / using a model
    f_zdot: ca.Function
    f_zdot_full: ca.Function
    f_znew: ca.Function

    # expected utility functions
    # 3D orientation matrix, inputs are (z,u), outputs are (R), a 3x3 matrix
    f_R: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # global thrust vector, inputs are (z,u), outputs are (Fi, Fj, Fk)
    f_T: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # body frame gravity forces
    f_Fg: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # global frame velocity
    f_vg: Callable[[np.ndarray, np.ndarray], np.ndarray]

    # casadi-compatible functions of same use
    ca_f_R: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ca_f_T: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ca_f_Fg: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ca_f_vg: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def step(self, state:RacerState):
        '''
        apply one step of the simulator to a racer state
        must be drone or point mass depending on model used.
        '''
        z, u = self.state2zu(state)
        zn = self.f_znew(z, u)
        zn = np.array(zn).squeeze()
        self.zu2state(state, zn, u)

    def get_rk4_dynamics(self, dt: float = None, use_mx: bool = False) -> ca.Function:
        '''
        get a function for fixed step rk4 integration of ODE
        MX is slower in python but is far better for code-gen
        '''
        if dt is None:
            dt = self.config.dt

        sym_class = ca.MX if use_mx else ca.SX

        z0 = sym_class.sym('z0',self.f_zdot.size_in(0))
        u0 = sym_class.sym('u0',self.f_zdot.size_in(1))

        k1 = self.f_zdot(z0          , u0)
        k2 = self.f_zdot(z0 + dt/2*k1, u0)
        k3 = self.f_zdot(z0 + dt/2*k2, u0)
        k4 = self.f_zdot(z0 + dt  *k3, u0)

        zn = z0 + dt / 6 * (k1 + k2*2 + k3*2 + k4)
        if isinstance(dt, sym_class):
            F = ca.Function('F', [z0, u0, dt], [zn])
        else:
            F = ca.Function('F', [z0, u0], [zn])
        return F

    def _setup(self):
        ''' general - purpose steps to set up dynamics model'''
        self._create_vars()
        self._compute_pose_evolution()
        self._compute_forces()
        self._compute_state_evolution()
        self._setup_integrator()
        self._setup_helper_functions()

    @abstractmethod
    def _create_vars(self):
        ''' create model variables, ie. state and inputs '''

    @abstractmethod
    def _compute_pose_evolution(self):
        ''' compute pose derivatives '''

    @abstractmethod
    def _compute_forces(self):
        ''' compute forces on the vehicle '''

    @abstractmethod
    def _compute_state_evolution(self):
        ''' compute overall state evolution '''

    def _setup_integrator(self):
        ''' set up integrator '''
        z = self._model_vars.z
        u = self._model_vars.u
        z_dot = self._model_vars.z_dot

        zmx = ca.MX.sym('z', z.size())
        umx = ca.MX.sym('u', u.size())

        self.f_zdot = ca.Function('zdot', [z, u], [z_dot])

        zdot_mx = self.f_zdot.call([zmx, umx])
        f_zdot_mx  = ca.Function('zdot', [zmx, umx], zdot_mx)
        zdot_mx  = f_zdot_mx(zmx, umx)

        self.f_znew = idas_integrator(zmx, umx, zdot_mx , self.config.dt)

    def _create_helper_function(self, name, inputs, expr)\
            -> Tuple[ca.Function, Callable]:
        '''
        utility for making helper functions
        '''
        f = ca.Function(name, inputs, expr)
        return f, lambda *inputs: np.array(f(*inputs)).squeeze()

    def _setup_helper_functions(self):
        z = self._model_vars.z
        u = self._model_vars.u
        helper_args = [z,u]

        # helper functions
        # rotation matrix
        R = self._model_vars.R
        self.ca_f_R, self.f_R = self._create_helper_function(
            'R',
            helper_args,
            [R])

        # total thrust
        T = self._model_vars.Tg
        self.ca_f_T, self.f_T = self._create_helper_function(
            'T',
            helper_args,
            [T])

        # force of gravity in body frame
        Fg = self._model_vars.Fgb
        self.ca_f_Fg, self.f_Fg = self._create_helper_function(
            'Fg',
            helper_args,
            [Fg])

        # velocity in global frame
        vg = self._model_vars.vg
        self.ca_f_vg, self.f_vg = self._create_helper_function(
            'vg',
            helper_args,
            [vg])

    @abstractmethod
    def get_empty_state(self) -> RacerState:
        ''' obtain an empty state compatible with the model'''

    @abstractmethod
    def state2u(self, state: RacerState) -> List[float]:
        ''' get input vector of a state for given model '''

    @abstractmethod
    def state2zu(self, state: RacerState) -> Tuple[List[float], List[float]]:
        ''' get state and input vector of a state for given model '''

    @abstractmethod
    def u2state(self, state: RacerState, u: List) -> None:
        ''' update racer input from vector'''

    @abstractmethod
    def du2state(self, state: RacerState, du: List) -> None:
        ''' update racer input rate from vector'''

    @abstractmethod
    def zu2state(self, state: RacerState, z: List, u: List) -> None:
        ''' update racer state and input from vectors'''

    @abstractmethod
    def zu(self) -> List[float]:
        ''' upper bound on state vector '''

    @abstractmethod
    def zl(self) -> List[float]:
        ''' lower bound on state vector '''

    @abstractmethod
    def uu(self) -> List[float]:
        ''' upper bound on input vector '''

    @abstractmethod
    def ul(self) -> List[float]:
        ''' lower bound on input vector '''

    @abstractmethod
    def duu(self) -> List[float]:
        ''' upper bound on input rate vector '''

    @abstractmethod
    def dul(self) -> List[float]:
        ''' lower bound on input rate vector '''

    @abstractmethod
    def add_model_stage_constraints(self, z, u, g, lbg, ubg):
        ''' add model stage constraints, ie. norm constraints on input '''


class ParametricDynamicsModel(DynamicsModel):
    ''' base class for dynamics involving a centerline '''
    line: BaseCenterline
    f_param_terms: ca.Function

    # parametric frame thrust vector, inputs are (z,u), outputs are (Fs, Fp, Fn)
    f_Tp: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ca_f_Tp: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def _setup_integrator(self):
        ''' set up ODE or DAE integrator for the model '''
        param_terms = self.line.sym_rep.param_terms
        f_param_terms = self.line.sym_rep.f_param_terms
        self.f_param_terms = f_param_terms

        z = self._model_vars.z
        u = self._model_vars.u
        z_dot = self._model_vars.z_dot

        zmx = ca.MX.sym('z', z.size())
        umx = ca.MX.sym('u', u.size())
        s_mx = zmx[0]

        f_zdot = ca.Function('zdot', [z, u, param_terms], [z_dot])
        self.f_param_terms = f_param_terms
        self.f_zdot_full = f_zdot

        zdot_mx = f_zdot.call([zmx, umx, f_param_terms(s_mx)])
        f_zdot_mx  = ca.Function('zdot', [zmx, umx], zdot_mx)
        zdot_mx  = f_zdot_mx (zmx, umx)
        self.f_zdot = f_zdot_mx

        self.f_znew = idas_integrator(zmx, umx, zdot_mx , self.config.dt)

    def _compute_forces(self):
        super()._compute_forces()
        R_rel = self._model_vars.R_rel
        Tb = self._model_vars.Tb

        Tp = R_rel @ Tb # parametric frame thrust

        self._model_vars.Tp = Tp

    def _create_helper_function(self, name, inputs, expr):
        '''
        utility for making helper functions
        it is assumed that inputs[0] is 'z' and z[:3] is the pose
        for filling in centerline parameters, which are assumed to be inputs[-1]

        this is isolated so that it can be overriden where vehicle model parameters
        are not constant, ie. sys id
        '''
        f_args = inputs[:-1]
        f = self.line.fill_in_centerline(name, expr, f_args)
        return f, lambda *f_args: np.array(f(*f_args)).squeeze()

    def _setup_helper_functions(self):
        z = self._model_vars.z
        u = self._model_vars.u
        param_terms = self.line.sym_rep.param_terms
        helper_args = [z,u, param_terms]

        # helper functions
        # rotation matrix
        R = self._model_vars.R
        self.ca_f_R, self.f_R = self._create_helper_function(
            'R',
            helper_args,
            [R])

        # total thrust
        T = self._model_vars.Tg
        self.ca_f_T, self.f_T = self._create_helper_function(
            'T',
            helper_args,
            [T])

        # total thrust in parametric frame
        Tp = self._model_vars.Tp
        self.ca_f_Tp, self.f_Tp = self._create_helper_function(
            'Tp',
            helper_args,
            [Tp])

        # force of gravity in body frame
        Fg = self._model_vars.Fgb
        self.ca_f_Fg, self.f_Fg = self._create_helper_function(
            'Fg',
            helper_args,
            [Fg])

        # velocity in global frame
        vg = self._model_vars.vg
        self.ca_f_vg, self.f_vg = self._create_helper_function(
            'vg',
            helper_args,
            [vg])

    def zu(self, s = 0):
        ''' upper bound on state vector '''
        zu = super().zu()
        zu[0] = self.line.s_max()
        zu[1] = self.line.y_max(s = s)
        zu[2] = self.line.n_max(s = s)
        return zu

    def zl(self, s = 0):
        ''' lower bound on state vector '''
        zl = super().zl()
        zl[0] = self.line.s_min()
        zl[1] = self.line.y_min(s = s)
        zl[2] = self.line.n_min(s = s)
        return zl


class InterpolatedDynamicsModel(DynamicsModel):
    '''
    dynamics model based on interpolated data
    intended for visualizing data from other projects
    use for simulation, racelines, etc... is not supported

    state vector:
        [xi, xj, xk, qi, qj, qk, qr]
    input vector:
        [vi, vj, vk, wi, wj, wk]
    (all of the above in global frame)
    '''
    def __init__(self):
        self._model_vars = DynamicsModelVars()
        self._setup()

    def step(self, state: RacerState):
        raise NotImplementedError('Cannot simulate from interpolated data ')

    def _create_vars(self):
        x = ca.SX.sym('x', 3)
        q = ca.SX.sym('q', 4)
        v = ca.SX.sym('v', 3)
        w = ca.SX.sym('w', 3)

        qi = q[0]
        qj = q[1]
        qk = q[2]
        qr = q[3]
        R = ca.vertcat(
            ca.horzcat(
                1 - 2*qj**2 - 2*qk**2,
                2*(qi*qj - qk*qr),
                2*(qi*qk + qj*qr)
            ),
            ca.horzcat(
                2*(qi*qj + qk*qr),
                1 - 2*qi**2 - 2*qk**2,
                2*(qj*qk - qi*qr),
            ),
            ca.horzcat(
                2*(qi*qk - qj*qr),
                2*(qj*qk + qi*qr),
                1 - 2*qi**2 - 2*qj**2
            )
        ) / (qi**2 + qj**2 + qk**2 + qr**2)

        T = [0, 0, 0]
        Fg = -9.81 * R[2,:]
        vg = v


        self._model_vars.z = ca.vertcat(x, q)
        self._model_vars.u = ca.vertcat(v, w)
        self._model_vars.R = R
        self._model_vars.Tg = T
        self._model_vars.Fgb = Fg
        self._model_vars.vg = vg

    def _compute_forces(self):
        pass

    def _compute_pose_evolution(self):
        pass

    def _compute_state_evolution(self):
        pass

    def _setup_integrator(self):
        pass

    def get_empty_state(self) -> RacerState:
        return RacerState()

    def state2u(self, state: RacerState):
        return np.concatenate([state.v.to_vec(), state.w.to_vec()])

    def state2zu(self, state: RacerState):
        z = np.concatenate([state.x.to_vec(), state.q.to_vec()])
        return z, self.state2u(state)

    def u2state(self, state: RacerState, u: List):
        state.v.from_vec(u[:3])
        state.w.from_vec(u[-3:])

    def du2state(self, state: RacerState, du: List):
        pass

    def zu2state(self, state: RacerState, z: List, u: List):
        state.x.from_vec(z[:3])
        state.q.from_vec(z[-4:])
        self.u2state(state, u)

    def zu(self):
        return [np.inf]

    def zl(self):
        return [-np.inf]

    def uu(self):
        return [np.inf]

    def ul(self):
        return [-np.inf]

    def duu(self):
        return [np.inf]

    def dul(self):
        return [-np.inf]

    def add_model_stage_constraints(self, z, u, g, lbg, ubg):
        return
