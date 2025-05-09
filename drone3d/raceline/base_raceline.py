'''
Standard methods and features for nonplanar raceline computation and manipulation
'''
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Callable
import time
import os

import numpy as np
import casadi as ca

from drone3d.utils.ca_utils import unpack_solution_helper
from drone3d.utils.discretization_utils import get_collocation_coefficients, \
    interpolate_collocation, get_intermediate_collocation_coefficients, \
    interpolate_linear

from drone3d.pytypes import PythonMsg, RacerConfig, RacerState
from drone3d.centerlines.base_centerline import BaseCenterline, GateShape
from drone3d.dynamics.dynamics_model import DynamicsModel, ParametricDynamicsModel
from drone3d.obstacles.mesh_obstacle import MeshObstacle, ObstacleFreeTube

from drone3d.visualization.objects import VertexObject, UBOObject


@dataclass
class RacelineConfig(PythonMsg):
    ''' configuration for a raceline solver '''
    verbose: bool = field(default = True)
    plot_iterations: bool = field(default = False)

    # number of collocation intervals and order thereof
    N: int = field(default = 30)
    K: int = field(default = 7)

    # flag to switch to rk4 direct method
    use_rk4: bool = field(default = False)

    # quadratic input penalty, if a float - multiplied by identity matrix
    R: np.ndarray  = field(default = 1e-7)
    dR: np.ndarray = field(default = 1e-7)

    # initial guesses for step size and speed
    h0: float = field(default = 1)
    v0: float = field(default = 1)

    #whether or not the raceline is periodic
    closed: bool = field(default = False)

    # force raceline to center of gates
    fix_gate_center: bool = field(default = False)

    # max IPOPT iterations
    max_iter:int  = field(default = 1000)

    # HSL linear solver to use (if installed)
    hsl_linear_solver: str = field(default = 'ma97')


@dataclass
class GlobalRacelineConfig(RacelineConfig):
    ''' config for raceline in global frame '''
    gate_xi: np.ndarray = field(default = None)
    gate_xj: np.ndarray = field(default = None)
    gate_xk: np.ndarray = field(default = None)


@dataclass
class ParametricRacelineConfig(RacelineConfig):
    ''' config for raceline in parametric frame '''
    # list of path lengths at which to add gate constraints
    # set to empty iterable to fix no gates
    # defaults to self.line.config.gate_s if not specified.
    # default will ignore last gate_s which equal line.s_max() for closed lines
    fixed_gates: List[float] = field(default = None)

    # force regularity constraints
    force_regularity: bool = field(default = True)


@dataclass
class RacelineResults(PythonMsg):
    ''' results template for raceline '''
    solve_time: float = field(default = None)
    ipopt_time: float = field(default = None)
    feval_time: float = field(default = None)
    feasible: bool = field(default = None)
    states: List[RacerState] = field(default = None)
    step_sizes: List[float] = field(default = None)
    time: float = field(default = None)
    periodic: bool = field(default = False)
    label: str = field(default = None)
    color: List[float] = field(default = None)
    z_interp: Callable[[float], np.ndarray] = field(default = None)
    u_interp: Callable[[float], np.ndarray] = field(default = None)
    du_interp: Callable[[float], np.ndarray] = field(default = None)
    global_frame: bool = field(default = None)


class BaseRaceline(ABC):
    ''' base raceline class '''
    line: BaseCenterline
    config: RacelineConfig
    model: DynamicsModel
    vehicle_config: RacerConfig

    # optional fields for warmstart
    ws_raceline: RacelineResults
    ws_model: DynamicsModel

    sym_class = ca.SX
    nlp: Dict[str, ca.SX] = {}

    setup_time: float = -1
    solve_time: float = -1
    ipopt_time: float = -1
    feval_time: float = -1

    solver: ca.nlpsol
    solver_w0: List[float]
    solver_ubw: List[float]
    solver_lbw: List[float]
    solver_ubg: List[float]
    solver_lbg: List[float]
    sol: Dict[str, ca.DM]

    z_interp: ca.Function
    u_interp: ca.Function
    du_interp: ca.Function

    f_sol_h: ca.Function
    f_sol_t: ca.Function
    f_sol_z: ca.Function
    f_sol_u: ca.Function
    f_sol_du: ca.Function

    global_frame:bool = None

    solve_callback_fig = None

    def __init__(self,
            line: BaseCenterline,
            config: RacelineConfig,
            vehicle_config: RacerConfig,
            ws_raceline: RacelineResults = None,
            ws_model: DynamicsModel = None):
        self.line = line
        self.config = config
        self.vehicle_config = vehicle_config
        self.ws_raceline = ws_raceline
        self.ws_model = ws_model
        self._setup()

        if self.solve_callback_fig is not None:
            self.solve_callback_fig.preview()

    def solve(self):
        ''' solve the raceline problem, return results'''
        t0 = time.time()
        solver_opts = {'x0': self.solver_w0,
                       'ubx': self.solver_ubw,
                       'lbx': self.solver_lbw,
                       'ubg': self.solver_ubg,
                       'lbg': self.solver_lbg}
        sol = self.solver(**solver_opts)
        self.solve_time = time.time() - t0
        self.sol = sol

        if self.config.verbose:
            sol_h = self.f_sol_h(sol['x'])
            print(f'reached target in    {np.sum(sol_h):0.3f} seconds')
            print(f'with a total cost of {float(sol["f"]):0.3f}')
            print(f'it took              {self.setup_time:0.3f} seconds to set up the problem')
            print(f'    and              {self.solve_time:0.3f} seconds to solve it')

        if self.config.plot_iterations:
            print('Warning - Timing statistics are incorrect when plotting iterations')
            while not self.solve_callback_fig.should_close:
                self.solve_callback_fig.draw()
            self.solve_callback_fig.close()

        stats = self.solver.stats()
        self.feval_time = \
            stats['t_wall_nlp_f'] + \
            stats['t_wall_nlp_g'] + \
            stats['t_wall_nlp_grad_f'] + \
            stats['t_wall_nlp_hess_l'] + \
            stats['t_wall_nlp_jac_g']
        self.ipopt_time = self.solve_time - self.feval_time

        return self._unpack_soln(sol)

    def get_ws(self):
        ''' get raceline warmstart, rather than solving'''
        self.solve_time = -1
        self.ipopt_time = -1
        self.feval_time = -1
        return self._unpack_soln({'x':self.solver_w0})

    @abstractmethod
    def _get_model(self, config: RacerConfig) -> DynamicsModel:
        ''' get a dynamics model a given surface '''
        return None

    @abstractmethod
    def _get_prediction_label(self) -> str:
        ''' get a string label for the prediction results'''
        return ''

    @abstractmethod
    def _get_prediction_color(self) -> List[float]:
        ''' get a RGBA color for the prediction results, fields are between 0 and 1'''
        return [1,0,0,1]

    def _ws_available(self) -> bool:
        return self.ws_model is not None and self.ws_raceline is not None

    def _setup(self):
        t0 = time.time()
        self.model = self._get_model(self.vehicle_config)
        self._setup_checks()
        self._create_nlp()
        self._create_solver()
        self.setup_time = time.time() - t0

    def _setup_checks(self):
        if self.config.use_rk4:
            self.config.h0 /= self.config.K
            self.config.N *= self.config.K
            self.config.K = 0

    def _create_nlp(self):
        self._initial_nlp_setup()
        self._assign_nlp_model()
        self._create_nlp_vars()
        self._enforce_model()
        self._add_gate_constraints()
        self._add_costs()
        self._create_problem()

    def _initial_nlp_setup(self):
        self.nlp = {}

        # nonlinear constraint vector
        self.nlp['g'] = []
        self.nlp['ubg'] = []
        self.nlp['lbg'] = []

        # cost
        self.nlp['J'] = 0

    @abstractmethod
    def _assign_nlp_model(self):
        ''' get a model for the dynamical system, populating self.nlp['f_ode']'''
        f_ode = self.nlp['f_ode']
        state_dim = f_ode.size_in(0)
        input_dim = f_ode.size_in(1)
        self.nlp['state_dim'] = state_dim
        self.nlp['input_dim'] = input_dim
        self._model_setup_checks()

    def _model_setup_checks(self):
        input_dim = self.nlp['input_dim']
        # if cost matrices have been provided as a scalar, convert to a matrix
        # done here since previously the correct dimensions are unknown
        if isinstance(self.config.R, (float, int)):
            self.config.R = np.eye(input_dim[0]) * self.config.R

        if isinstance(self.config.dR, (float, int)):
            self.config.dR = np.eye(input_dim[0]) * self.config.dR

    def _eval_ode(self, n, k=0):
        f_ode = self.nlp['f_ode']
        Z = self.nlp['Z']
        U = self.nlp['U']

        return f_ode(Z[n,k], U[n,k])

    def _create_nlp_vars(self):
        N = self.config.N
        K = self.config.K

        if not self.config.use_rk4:
            tau, B, C, D = get_collocation_coefficients(K)

            self.nlp['tau'] = tau
            self.nlp['B'] = B
            self.nlp['C'] = C
            self.nlp['D'] = D

        var_shape = (N,K+1)
        H  = np.resize(np.array([], dtype = self.sym_class), (N))        # interval step size
        T  = np.resize(np.array([], dtype = self.sym_class), var_shape)  # time from start
        Z  = np.resize(np.array([], dtype = self.sym_class), var_shape)  # differential state
        U  = np.resize(np.array([], dtype = self.sym_class), var_shape)  # input
        dU = np.resize(np.array([], dtype = self.sym_class), var_shape)  # input rate

        # step size variables
        for n in range(N):
            hk = self.sym_class.sym(f'h_{n}')
            H[n] = hk

        # other variables
        T[0,0] = 0
        for n in range(N):
            if n > 0:
                T[n,0] = T[n-1, 0] + H[n-1]
            for k in range(K+1):
                if k > 0:
                    T[n,k] = self.nlp['tau'][k] * H[n] + T[n,0]
                Z[n,k]  = self.sym_class.sym(f'z_{n}_{k}',  self.nlp['state_dim'])
                U[n,k]  = self.sym_class.sym(f'u_{n}_{k}',  self.nlp['input_dim'])
                dU[n,k] = self.sym_class.sym(f'du_{n}_{k}', self.nlp['input_dim'])


        self.nlp['H'] = H
        self.nlp['T'] = T
        self.nlp['Z'] = Z
        self.nlp['U'] = U
        self.nlp['dU'] = dU

    def _zF(self) -> ca.SX:
        ''' last state of last interval '''
        if self.config.use_rk4:
            h = self.nlp['H'][-1]
            z = self.nlp['Z'][-1,0]
            u = self.nlp['U'][-1,0]
            F = self.model.get_rk4_dynamics(h, False)
            return self._state_continuity_operator(F(z, u, h))
        K = self.config.K
        Z = self.nlp['Z']
        D = self.nlp['D']
        zF = 0
        for k in range(K+1):
            zF += Z[-1,k] * D[k]
        return self._state_continuity_operator(zF)

    def _uF(self) -> ca.SX:
        ''' last input of last interval '''
        if self.config.use_rk4:
            return self.nlp['U'][-1,0] + self.nlp['dU'][-1,0] * self.nlp['H'][-1]
        K = self.config.K
        U = self.nlp['U']
        D = self.nlp['D']
        uF = 0
        for k in range(K+1):
            uF += U[-1,k] * D[k]
        return uF

    def _enforce_model(self):
        for n in range(self.config.N):
            if not self.config.use_rk4:
                self._enforce_collocation_interval(n)
            else:
                self._enforce_rk4_interval(n)

        if self.config.closed:
            self._enforce_loop_closure()
        else:
            self._enforce_initial_constraints()
            self._enforce_terminal_constraints()

    def _enforce_rk4_interval(self, n):
        if n == self.config.N - 1:
            return
        h = self.nlp['H'][n]
        F = self.model.get_rk4_dynamics(h, False)

        Z = self.nlp['Z']
        U = self.nlp['U']
        dU = self.nlp['dU']
        z = Z[n,0]
        u = U[n,0]
        zn = self._state_continuity_operator(F(z, u, h))
        un = U[n,0] + dU[n,0]*h / 2

        state_dim = self.nlp['state_dim']
        input_dim = self.nlp['input_dim']

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        g += [Z[n+1,0] - zn]
        g += [U[n+1,0] - un]
        ubg += [0.] * (state_dim[0] + input_dim[0])
        lbg += [0.] * (state_dim[0] + input_dim[0])

        self.model.add_model_stage_constraints(
            z, u, g, lbg, ubg
        )

    def _enforce_collocation_interval(self, n):
        self._enforce_collocation_interval_ode(n)
        self._enforce_collocation_interval_constraints(n)
        self._enforce_collocation_interval_continuity(n)

    def _enforce_collocation_interval_ode(self, n):
        K = self.config.K
        C = self.nlp['C']

        H = self.nlp['H']
        Z = self.nlp['Z']
        U = self.nlp['U']
        dU = self.nlp['dU']

        state_dim = self.nlp['state_dim']

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        for k in range(K+1):
            poly_ode = 0
            poly_du = 0
            for k2 in range(K+1):
                poly_ode += C[k2][k] * Z[n,k2] / H[n]
                poly_du  += C[k2][k] * U[n,k2] / H[n]

            func_ode = self._eval_ode(n,k)

            if isinstance(self.model, ParametricDynamicsModel):
                g += [poly_ode[0]]
                ubg += [np.inf]
                lbg += [0]

            if k > 0:
                g += [func_ode - poly_ode]
                ubg += [0.] * state_dim[0]
                lbg += [0.] * state_dim[0]

            g += [dU[n,k] - poly_du]
            ubg += [0.] * dU[n,k].shape[0]
            lbg += [0.] * dU[n,k].shape[0]

    def _enforce_collocation_interval_constraints(self, n):
        K = self.config.K
        Z = self.nlp['Z']
        U = self.nlp['U']

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        for k in range(K+1):
            self.model.add_model_stage_constraints(
                Z[n,k],
                U[n,k],
                g,
                lbg,
                ubg)

    def _state_continuity_operator(self, z):
        '''
        any additional operations to do on states between intervals
        ex: quaternion normalization
        '''
        return z

    def _enforce_collocation_interval_continuity(self, n):
        K = self.config.K
        D = self.nlp['D']

        Z = self.nlp['Z']
        U = self.nlp['U']

        state_dim = self.nlp['state_dim']
        input_dim = self.nlp['input_dim']

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        # add continuity from previous interval
        poly_prev_state = 0
        poly_prev_input = 0
        if n >= 1:
            for k in range(K+1):
                poly_prev_state += Z[n-1,k] * D[k]
                poly_prev_input += U[n-1,k] * D[k]

            poly_prev_state = self._state_continuity_operator(poly_prev_state)

            g += [Z[n,0] - poly_prev_state]
            ubg += [0.] * state_dim[0]
            lbg += [0.] * state_dim[0]

            g += [U[n,0] - poly_prev_input]
            ubg += [0.] * input_dim[0]
            lbg += [0.] * input_dim[0]

    def _enforce_loop_closure(self):
        Z = self.nlp['Z']
        U = self.nlp['U']

        state_dim = self.nlp['state_dim']
        input_dim = self.nlp['input_dim']

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        z0 = Z[0,0]
        u0 = U[0,0]
        zF = self._zF()
        uF = self._uF()
        zF = self._state_continuity_operator(zF)

        g += [uF - u0]
        ubg += [0.] * input_dim[0]
        lbg += [0.] * input_dim[0]
        g += [zF - z0]
        ubg += [0.] * state_dim[0]
        lbg += [0.] * state_dim[0]

    def _enforce_initial_constraints(self):
        ''' initial constraints, zero velocity by default '''
        z = self.nlp['Z'][0,0]
        u = self.nlp['U'][0,0]
        vg = self.model.ca_f_vg(z, u)

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        g += [vg.T @ vg]
        ubg += [0.]
        lbg += [-np.inf]

    def _enforce_terminal_constraints(self):
        ''' terminal constraints, zero velocity by default'''
        zF = self._zF()
        uF = self._uF()

        vg = self.model.ca_f_vg(zF, uF)

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        g += [vg.T @ vg]
        ubg += [0.]
        lbg += [-np.inf]

    def _fix_gate(self, x_var: ca.SX, s: float,
            include_axial_fix: bool = False):
        ''' add constraints for a single gate
        requires global frame position variable and 's' coordinate for
        where the gate should be

        include_axial_fix should be True for global frame problems.
        '''
        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']
        gate_x = self.line.gate_position(s)
        if self.config.fix_gate_center:
            g += [x_var - gate_x]
            ubg += [0]*3
            lbg += [0]*3

        else:
            R = self.line.gate_orientation(s)

            if self.line.config.gate_shape == GateShape.CIRCLE:
                e1 = R[:,0]
                e2 = R[:,1]
                e3 = R[:,2]

                r_sq = ((x_var-gate_x).T @ e2)**2 +  ((x_var-gate_x).T @ e3)**2
                r_sq_max = (self.line.config.gate_ri - self.model.config.collision_radius)**2

                g += [r_sq]
                ubg += [r_sq_max]
                lbg += [-np.inf]

                if include_axial_fix:
                    g += [x_var.T @ e1 - gate_x.T @ e1]
                    ubg += [0.]
                    lbg += [0.]

            elif self.line.config.gate_shape == GateShape.SQUARE:
                delta = R.T @ (x_var - gate_x)
                d_max = self.line.config.gate_ri - self.model.config.collision_radius

                if include_axial_fix:
                    g += [delta]
                    ubg += [0., d_max, d_max]
                    lbg += [0., -d_max, -d_max]
                else:
                    g += [delta[1:]]
                    ubg += [d_max, d_max]
                    lbg += [-d_max, -d_max]
            else:
                raise NotImplementedError('Unhandled Gate Shape')

    @abstractmethod
    def _add_gate_constraints(self):
        ''' add any constraints for gates to pass through '''

    def _add_costs(self):
        N = self.config.N
        K = self.config.K
        for n in range(N):
            for k in range(K+1):
                self._add_stage_cost(n,k)

    def _add_stage_cost(self, n, k = 0):
        H = self.nlp['H']
        if self.config.use_rk4:
            self.nlp['J'] += self._stage_cost(n, k) * H[n]
        else:
            B = self.nlp['B']
            self.nlp['J'] += self._stage_cost(n, k) * H[n] * B[k]

    def _stage_cost(self, n, k):
        ''' stage cost, time plus regularization '''
        U = self.nlp['U']
        dU = self.nlp['dU']

        return ca.bilin(self.config.R, U[n,k], U[n,k]) + \
               ca.bilin(self.config.dR, dU[n,k], dU[n,k]) + \
               1

    def _create_problem(self):
        H = self.nlp['H']
        T = self.nlp['T']
        Z = self.nlp['Z']
        U = self.nlp['U']
        dU = self.nlp['dU']

        w, w0, ubw, lbw = self._build_decision_vector()

        g = ca.vertcat(*self.nlp['g'])
        w = ca.vertcat(*w)

        self.nlp['g'] = g
        self.nlp['w'] = w
        self.nlp['ubw'] = ubw
        self.nlp['lbw'] = lbw

        self.solver_w0 = w0
        self.solver_ubw = ubw
        self.solver_lbw = lbw
        self.solver_ubg = self.nlp['ubg']
        self.solver_lbg = self.nlp['lbg']

        if self.config.use_rk4:
            self.z_interp = interpolate_linear(w, H, Z)
            self.u_interp = interpolate_linear(w, H, U)
            self.du_interp = interpolate_linear(w, H, dU)
        else:
            self.z_interp = interpolate_collocation(w, H, Z, self.config)
            self.u_interp = interpolate_collocation(w, H, U, self.config)
            self.du_interp = interpolate_collocation(w, H, dU, self.config)

        def packer(X):
            '''
            function to pack problem data into CasADi format for
            unpacking helper functions (Tested 18.04+)
            '''
            return ca.horzcat(*[ca.horzcat(*X[k]) for k in range(X.shape[0])]).T

        self.f_sol_h  = unpack_solution_helper('h' , w, H.tolist())
        self.f_sol_t  = unpack_solution_helper('h' , w, [packer(T)])
        self.f_sol_z  = unpack_solution_helper('z' , w, [packer(Z)])
        self.f_sol_u  = unpack_solution_helper('u' , w, [packer(U)])
        self.f_sol_du = unpack_solution_helper('du', w, [packer(dU)])

    def _build_decision_vector(self):
        N = self.config.N
        K = self.config.K
        H = self.nlp['H']

        w = []
        w0 = []
        ubw = []
        lbw = []

        # build up variable vector
        for n in range(N):
            w += [H[n]]
            h0 = self._guess_h(n)
            ubw += [h0*10]
            lbw += [h0/100]
            w0 += [h0]

        for n in range(N):
            for k in range(0, K+1):
                self._add_stage_decision_variables(n, k, w, w0, ubw, lbw)

        return w, w0, ubw, lbw

    def _add_stage_decision_variables(self, n, k, w, w0, ubw, lbw):
        state_u, state_l = self._get_state_bounds(n,k)
        input_u, input_l = self._get_input_bounds(n,k)
        input_du, input_dl = self._get_input_rate_bounds(n,k)

        Z = self.nlp['Z']
        U = self.nlp['U']
        dU = self.nlp['dU']

        w += [Z[n,k]]
        lbw += state_l
        ubw += state_u

        w += [U[n,k]]
        lbw += input_l
        ubw += input_u

        w += [dU[n,k]]
        lbw += input_dl
        ubw += input_du

        w0 += self._guess_z(n,k)
        w0 += self._guess_u(n,k)
        w0 += self._guess_du(n,k)

    def _get_state_bounds(self,n,k):
        # pylint: disable=unused-argument
        return self.model.zu(), self.model.zl()

    def _get_input_bounds(self,n,k):
        # pylint: disable=unused-argument
        return self.model.uu(), self.model.ul()

    def _get_input_rate_bounds(self,n,k):
        # pylint: disable=unused-argument
        return self.model.duu(), self.model.dul()

    def _guess_h(self, n):
        if self.ws_model and self.ws_raceline:
            return self.ws_raceline.step_sizes[n]
        if self.config.h0:
            return self.config.h0
        return 1

    @abstractmethod
    def _guess_z(self, n, k):
        ''' guess state vector '''

    def _guess_u(self,n,k):
        # pylint: disable=unused-argument
        input_dim = self.nlp['input_dim']
        return [0.] * input_dim[0]

    def _guess_du(self,n,k):
        # pylint: disable=unused-argument
        input_dim = self.nlp['input_dim']
        return [0.] * input_dim[0]

    def _create_solver(self):
        prob = {}
        prob['x'] = self.nlp['w']
        prob['g'] = self.nlp['g']
        prob['f'] = self.nlp['J']
        if 'p' in self.nlp:
            prob['p'] = self.nlp['p']

        if self.config.verbose:
            opts = {'ipopt.sb':'yes'}
        else:
            opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}

        if os.path.exists('/usr/local/lib'):
            if 'libcoinhsl.so' in os.listdir('/usr/local/lib/') and \
                    '3.6' in ca.__version__:
                # hsllib option is only supported on newer ipopt versions
                opts['ipopt.linear_solver'] = self.config.hsl_linear_solver
                opts['ipopt.hsllib'] = '/usr/local/lib/libcoinhsl.so'
            elif 'libhsl.so' in os.listdir('/usr/local/lib/'):
                # check for obsolete hsl install and that it is on search path
                if '/usr/local/lib' in os.environ['LD_LIBRARY_PATH']:
                    opts['ipopt.linear_solver'] = self.config.hsl_linear_solver
                else:
                    print('ERROR - HSL is present but not on path')
                    print('Run "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/"')
                    print('Defaulting to MUMPS')

        if '3.6' in ca.__version__ and 'ipopt.linear_solver' not in opts:
            print('WARNING - Casadi 3.6.X default linear solver for IPOPT is unreliable')
            print('Downgrading or using HSL solvers will be necessary for drone racelines')

        opts['ipopt.max_iter'] = self.config.max_iter
        if '3.6' in ca.__version__:
            opts['ipopt.timing_statistics'] = 'yes'
        opts['ipopt.honor_original_bounds'] = 'yes'

        if self.config.plot_iterations:
            # pylint: disable=import-outside-toplevel
            # disabled to avoid circular import and since speed is not critical if using this
            from drone3d.visualization.iteration_callback_fig import CallbackWindow
            callback = CallbackWindow(self.line, prob,
                unpacker = lambda sol: BaseRaceline._unpack_soln(self, sol),
                solver = self)
            self.solve_callback_fig = callback
            opts['iteration_callback'] = callback

        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    def _unpack_soln(self, sol):
        '''
        unpacks the planned solution:
            state at each interval (including normal forces, global pose, etc..)
            interpolation objects for state, input, input rate, and dae state (if applicable)

        for fast online control, unpacking the state may be undesirable,
        as this can take ~ 1ms per state even for simple models
        '''
        sol_h = self.f_sol_h(sol['x'])
        sol_t = self.f_sol_t(sol['x'])
        sol_z = self.f_sol_z(sol['x'])
        sol_u = self.f_sol_u(sol['x'])
        sol_du = self.f_sol_du(sol['x'])

        states = []
        for t, z, u, du in zip(sol_t, sol_z, sol_u, sol_du):
            state = self.model.get_empty_state()
            state.t = t
            self.model.zu2state(state, z, u) #this is a current bottleneck
            self.model.du2state(state, du)
            states.append(state)

        t = self.sym_class.sym('t')

        z_interp = self.z_interp.call([t, sol['x']])
        z_interp = ca.Function('z_interp', [t], z_interp)
        def z_interp_np(t):
            ''' unpacking state as a numpy array'''
            return np.array(z_interp(t)).squeeze()

        u_interp = self.u_interp.call([t, sol['x']])
        u_interp = ca.Function('u_interp', [t], u_interp)
        def u_interp_np(t):
            ''' unpacking input as a numpy array'''
            return np.array(u_interp(t)).squeeze()

        du_interp = self.du_interp.call([t, sol['x']])
        du_interp = ca.Function('du_interp', [t], du_interp)
        def du_interp_np(t):
            ''' unpacking input rate as a numpy array'''
            return np.array(du_interp(t)).squeeze()

        if hasattr(self, 'solver'):
            feasible = self.solver.stats()['success']
        else:
            # catches startup case for callback figure
            feasible = False

        return RacelineResults(
            states = states,
            step_sizes = sol_h,
            time = np.sum(sol_h),
            periodic = self.config.closed,
            label = self._get_prediction_label(),
            color = self._get_prediction_color(),
            z_interp = z_interp_np,
            u_interp = u_interp_np,
            du_interp = du_interp_np,
            solve_time = self.solve_time,
            feval_time = self.feval_time,
            ipopt_time = self.ipopt_time,
            feasible = feasible,
            global_frame = self.global_frame)


class BaseGlobalRaceline(BaseRaceline):
    ''' base global frame raceline class '''
    config: GlobalRacelineConfig
    gate_n_interval: int
    global_frame = True

    def _setup_checks(self):
        super()._setup_checks()
        # fit a spline to gates for warmstarting position
        x = np.array([self.config.gate_xi, self.config.gate_xj, self.config.gate_xk])
        if self.config.closed:
            if not (x[:,0] == x[:,-1]).all():
                x = np.hstack([x, x[:,0:1]])

        # round N up to be a multiple of number of gates.
        num_gates = x.shape[1]
        num_phases = num_gates -1
        self.config.N = int(num_phases * np.ceil(self.config.N / num_phases))
        self.gate_n_interval = int(self.config.N / num_phases)

    def _assign_nlp_model(self):
        self.nlp['f_ode'] = self.model.f_zdot
        return super()._assign_nlp_model()

    def _enforce_model(self):
        # constrain all time stpes between gates to be identical
        H = self.nlp['H']
        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']
        for n in range(0, self.config.N, self.gate_n_interval):
            h1 = H[n]
            for n2 in range(n+1, n+self.gate_n_interval):
                h2 = H[n2]
                g += [h2 - h1]
                ubg += [0.]
                lbg += [0.]

        return super()._enforce_model()

    def _add_gate_constraints(self):

        Z = self.nlp['Z']

        for gate_no, n in enumerate(range(0, self.config.N, self.gate_n_interval)):
            self._fix_gate(Z[n,0][:3], gate_no, include_axial_fix=True)

        # final gate if not closed
        if not self.config.closed:
            zF = self._zF()
            gate_no = len(self.config.gate_xi) -1
            self._fix_gate(zF[:3], gate_no, include_axial_fix=True)

    def _guess_z(self, n, k):
        if self.config.use_rk4:
            gate_no = n / self.gate_n_interval
        else:
            gate_no = (n + k / self.config.K) / self.gate_n_interval
        x_guess = self.line.p2xc(gate_no)
        v_guess = self.line.p2es(gate_no)
        v_guess = v_guess / np.linalg.norm(v_guess) * self.config.v0

        state_dim = self.nlp['state_dim']
        z = [0.] * state_dim[0]
        z[0] = x_guess[0]
        z[1] = x_guess[1]
        z[2] = x_guess[2]
        z[3] = v_guess[0]
        z[4] = v_guess[1]
        z[5] = v_guess[2]
        return z


class BaseParametricRaceline(BaseRaceline):
    ''' base parametric raceline class '''
    config: ParametricRacelineConfig
    model: ParametricDynamicsModel
    global_frame = False

    def _setup_checks(self):
        if not self.line.cleanly_closed:
            # currently needed to make sure orientation and velocity
            # are cleanly joined.
            # see self._enforce_loop_closure
            if not self.model.config.global_r:
                raise NotImplementedError(
                    'Global orientation must be used for skewly closed centerlines')
        return super()._setup_checks()

    def _assign_nlp_model(self):
        self.nlp['f_ode'] = self.model.f_zdot_full

        param_dim = self.nlp['f_ode'].size_in(2)
        self.nlp['param_dim'] = param_dim
        return super()._assign_nlp_model()

    def _eval_ode(self, n, k = 0):
        f_ode = self.nlp['f_ode']
        Z = self.nlp['Z']
        U = self.nlp['U']

        s = self._get_s(n, k)
        param_terms = self.model.f_param_terms(s)
        return f_ode(Z[n,k], U[n,k], param_terms)

    def _get_s(self, n, k):
        '''
        return fixed "s" coordinate position of a collocation point
        in interval n for the kth collocation point for fixed space planning
        '''
        ds = (self.line.s_max() - self.line.s_min()) / self.config.N

        if self.config.use_rk4:
            return self.line.s_min() + ds * n

        if 'tau' not in self.nlp:
            self.nlp['tau'], _, _, _ = get_collocation_coefficients(self.config.K)
        return self.line.s_min() + ds * (n + self.nlp['tau'][k])

    def _add_gate_constraints(self):
        fixed_gates = self.config.fixed_gates
        if fixed_gates is None:
            if self.line.config.gate_s is not None:
                fixed_gates = self.line.config.gate_s
                # remove final gate if periodic
                if self.line.s_min() in fixed_gates:
                    if self.line.config.closed and self.config.closed:
                        fixed_gates = np.array([k for k in fixed_gates if k != self.line.s_max()])
            else:
                return

        for s in fixed_gates:
            s0 = self._get_s(0,0)
            if s < s0:
                raise TypeError('Gate is before start')
            n = 0
            while not self._get_s(n+1, 0) > s:
                n +=1
                s0 = self._get_s(n,0)
                if n == self.config.N:
                    if s > s0 + 0.1:
                        raise TypeError('Gate is after end')

            if n == self.config.N:
                z_gate = self._zF()
            else:
                sf = self._get_s(n+1, 0)
                # fraction along the interval to add gate at
                d = (s - s0) / (sf - s0)
                # state at gate
                Z = self.nlp['Z']
                if self.config.use_rk4:
                    z_gate = Z[n,0] + d * (Z[n+1,0] - Z[n,0])
                else:
                    D = get_intermediate_collocation_coefficients(
                        self.config.K, d)

                    z_gate = 0
                    for k in range(self.config.K+1):
                        z_gate += Z[n,k] * D[k]

            x_gate = self.line.p2xc(s) \
                + z_gate[1] * self.line.p2ey(s) \
                + z_gate[2] * self.line.p2en(s)

            self._fix_gate(x_gate, s, include_axial_fix=False)

    def _zF(self) -> ca.SX:
        if self.config.use_rk4:
            h = self.nlp['H'][-1]
            z = self.nlp['Z'][-1,0]
            u = self.nlp['U'][-1,0]

            s = self._get_s(self.config.N-1, 0)
            param_terms = self.model.f_param_terms(s)
            k1 = self.model.f_zdot_full(z,          u, param_terms)
            k2 = self.model.f_zdot_full(z + h/2*k1, u, param_terms)
            k3 = self.model.f_zdot_full(z + h/2*k2, u, param_terms)
            k4 = self.model.f_zdot_full(z + h  *k3, u, param_terms)

            zn = z + h / 6 * (k1 + k2*2 + k3*2 + k4)

            return self._state_continuity_operator(zn)
        return super()._zF()

    def _enforce_rk4_interval(self, n):
        # path length at the start of the interval
        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']
        Z = self.nlp['Z']

        ss = self._get_s(n, 0)
        g += [Z[n,0][0] - ss]
        ubg += [0.]
        lbg += [0.]

        if n == self.config.N - 1:
            return

        h = self.nlp['H'][n]

        U = self.nlp['U']
        dU = self.nlp['dU']
        z = Z[n,0]
        u = U[n,0]

        s = self._get_s(n, 0)
        param_terms = self.model.f_param_terms(s)
        k1 = self.model.f_zdot_full(z,          u, param_terms)
        k2 = self.model.f_zdot_full(z + h/2*k1, u, param_terms)
        k3 = self.model.f_zdot_full(z + h/2*k2, u, param_terms)
        k4 = self.model.f_zdot_full(z + h  *k3, u, param_terms)

        zn = z + h / 6 * (k1 + k2*2 + k3*2 + k4)
        zn = self._state_continuity_operator(zn)
        un = U[n,0] + dU[n,0]*h / 2

        state_dim = self.nlp['state_dim']
        input_dim = self.nlp['input_dim']

        g += [Z[n+1,0][1:] - zn[1:]]
        g += [U[n+1,0] - un]
        ubg += [0.] * (state_dim[0] + input_dim[0] - 1)
        lbg += [0.] * (state_dim[0] + input_dim[0] - 1)

        # path length at the end of the rk4 pass
        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']
        g += [zn[0] - self._get_s(n+1,0)]
        ubg += [0.]
        lbg += [0.]

        self.model.add_model_stage_constraints(
            z, u, g, lbg, ubg
        )

        # regularity constraints
        if self.config.force_regularity:
            ky = self.line.p2ky(self._get_s(n, 0))
            kn = self.line.p2kn(self._get_s(n, 0))
            if ky**2 + kn**2 > 0.1:
                g += [kn*Z[n,0][1] - ky*Z[n,0][2]]
                ubg += [self.line.config.gamma]
                lbg += [-np.inf]

    def _enforce_collocation_interval_constraints(self, n):
        K = self.config.K
        Z = self.nlp['Z']
        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        # regularity constraints
        if self.config.force_regularity:
            for k in range(K+1):
                ky = self.line.p2ky(self._get_s(n, k))
                kn = self.line.p2kn(self._get_s(n, k))
                if ky**2 + kn**2 > 0.1:
                    g += [kn*Z[n,k][1] - ky*Z[n,k][2]]
                    ubg += [self.line.config.gamma]
                    lbg += [-np.inf]
        return super()._enforce_collocation_interval_constraints(n)

    def _enforce_collocation_interval_continuity(self, n):
        K = self.config.K
        D = self.nlp['D']

        Z = self.nlp['Z']
        U = self.nlp['U']

        state_dim = self.nlp['state_dim']
        input_dim = self.nlp['input_dim']

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        # add continuity from previous interval
        poly_prev_state = 0
        poly_prev_input = 0
        if n >= 1:
            for k in range(K+1):
                poly_prev_state += Z[n-1,k] * D[k]
                poly_prev_input += U[n-1,k] * D[k]

            poly_prev_state = self._state_continuity_operator(poly_prev_state)

            # path length of the start of each interval is constrained by spacing constraints
            g += [Z[n,0][1:] - poly_prev_state[1:]]
            ubg += [0.] * (state_dim[0]-1)
            lbg += [0.] * (state_dim[0]-1)

            g += [U[n,0] - poly_prev_input]
            ubg += [0.] * input_dim[0]
            lbg += [0.] * input_dim[0]

        # fixed space constraints
        # state at the end of the interval
        zN = 0
        for k in range(K+1):
            zN += Z[n,k] * D[k]

        # path length at the start of the interval
        ss = self._get_s(n, 0)
        g += [Z[n,0][0] - ss]
        ubg += [0.]
        lbg += [0.]

        # path length at the end of the interval
        sf = self._get_s(n+1, 0)
        g += [zN[0] - sf]
        ubg += [0.]
        lbg += [0.]

    def _enforce_loop_closure(self):
        Z = self.nlp['Z']
        U = self.nlp['U']

        state_dim = self.nlp['state_dim']
        input_dim = self.nlp['input_dim']

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        z0 = Z[0,0]
        u0 = U[0,0]
        zF = self._zF()
        uF = self._uF()
        zF = self._state_continuity_operator(zF)

        g += [uF - u0]
        ubg += [0.] * input_dim[0]
        lbg += [0.] * input_dim[0]
        if self.line.cleanly_closed:
            g += [zF[1:] - z0[1:]]
            ubg += [0.] * (state_dim[0]-1)
            lbg += [0.] * (state_dim[0]-1)

        else:
            ey1 = self.line.p2ey(self.line.s_min())
            en1 = self.line.p2en(self.line.s_min())
            ey2 = self.line.p2ey(self.line.s_max()-0.001)
            en2 = self.line.p2en(self.line.s_max()-0.001)

            A = np.array([
                [ey1 @ ey2, en1 @ ey2],
                [ey1 @ en2, en1 @ en2]
            ])
            # position continuity
            res = A @ z0[1:3] - zF[1:3]
            g += [res]
            ubg += [0., 0.]
            lbg += [0., 0.]

            # orientation continuity
            g += [z0[3:] - zF[3:]]
            ubg += [0.] * (state_dim[0] - 3)
            lbg += [0.] * (state_dim[0] - 3)

    def _get_state_bounds(self,n,k):
        return self.model.zu(self._get_s(n,k)), self.model.zl(self._get_s(n,k))

    def _guess_h(self, n):
        if self.ws_model and self.ws_raceline:
            return self.ws_raceline.step_sizes[n]
        elif self.config.h0:
            return self.config.h0
        ds = (self.line.s_max() - self.line.s_min()) / self.config.N
        return ds / self.config.v0 * self.line.p2mag_xcs(ds * n)

    def _guess_z(self,n,k):
        state_dim = self.nlp['state_dim']
        z = [0.] * state_dim[0]
        z[0] = self._get_s(n,k)
        if not self.model.config.global_r:
            z[3] = self.config.v0
        else:
            v = self.config.v0 * self.line.p2es(self._get_s(n,k))
            z[3] = v[0]
            z[4] = v[1]
            z[5] = v[2]
        return z


class BaseParametricObstacleRaceline(BaseParametricRaceline):
    '''
    parametric raceline with obstacle avoidance with resepct to a
    mesh environment
    '''
    mesh_obstacle: MeshObstacle
    tube: ObstacleFreeTube = None
    calc_tube_time: float = -1
    setup_info: dict

    def __init__(self,
            line: BaseCenterline,
            config: ParametricRacelineConfig,
            vehicle_config: RacerConfig,
            mesh_obstacle: MeshObstacle,
            tube: ObstacleFreeTube = None,
            ws_raceline: RacelineResults = None,
            ws_model: DynamicsModel = None):
        self.mesh_obstacle = mesh_obstacle
        self.tube = tube

        super().__init__(
            line = line,
            config = config,
            vehicle_config = vehicle_config,
            ws_raceline = ws_raceline,
            ws_model = ws_model
        )

    def _setup(self):
        # subtract tube compute time if it is set
        super()._setup()
        if self.calc_tube_time > 0 :
            self.setup_time -= self.calc_tube_time

    def _add_gate_constraints(self):
        super()._add_gate_constraints()
        self._add_mesh_constraints()

    def _add_mesh_constraints(self):
        if self.tube is None:
            s = np.array([self._get_s(n,k)
                for n in range(self.config.N)
                for k in range(self.config.K+1)])
            t0 = time.time()
            self.tube = self.mesh_obstacle.compute_plannning_tube(
                self.line, s, self.model.config.collision_radius)
            self.calc_tube_time = time.time() - t0

        Z = self.nlp['Z']
        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        for n in range(self.config.N):
            for k in range(self.config.K+1):
                self.tube.add_constraints_parametric(self._get_s(n,k), Z[n,k], g, ubg, lbg)

    def _unpack_soln(self, sol):
        raceline = super()._unpack_soln(sol)

        x = np.array([state.x.to_vec() for state in raceline.states])
        distances = self.mesh_obstacle.signed_distance(x)

        for state, d in zip(raceline.states, distances):
            state.d = d

        if self.config.verbose:
            # pylint: disable=line-too-long
            if distances.min() >= self.model.config.collision_radius:
                print(f'Passed Collision Test (min: {distances.min():0.3f}m, max: {distances.max():0.3f}m, pass: {self.model.config.collision_radius:0.3f}m)')
            else:
                print(f'Failed Collision Test (min: {distances.min():0.3f}m, max: {distances.max():0.3f}m, pass: {self.model.config.collision_radius:0.3f}m)')
        return raceline

    def triangulate_setup_info(self, ubo: UBOObject) -> Dict[str, VertexObject]:
        ''' generate drawables for setup info '''
        return self.tube.get_vertex_objects(ubo)
