''' dynamics models of drones '''
import casadi as ca
import numpy as np

from drone3d.pytypes import DroneConfig, DroneState, GlobalQuaternion, GlobalEulerAngles,\
    RelativeQuaternion, RelativeEulerAngles
from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.dynamics.rotations import Rotation, Reference, Parameterization
from drone3d.dynamics.dynamics_model import DynamicsModel, \
    ParametricDynamicsModel, DynamicsModelVars

class DroneModel(DynamicsModel):
    ''' inertial frame drone model '''
    config: DroneConfig
    rot: Rotation

    # keep track of last global quaternion if using euler angles or parametric orientaiton
    # to help keep things continuous
    _last_q: np.ndarray = None

    def __init__(self,
            config: DroneConfig):
        self.config = config
        self._model_vars = DynamicsModelVars()
        self._setup()

    def _create_vars(self):
        if self.config.use_quat:
            _param = Parameterization.ESP
        else:
            _param = Parameterization.YPR
        if self.config.global_r:
            _ref=  Reference.GLOBAL
        else:
            _ref = Reference.PARAMETRIC
        self.rot = Rotation(
            ref = _ref,
            param = _param
        )

        self._model_vars.p = ca.SX.sym('x', 3)
        self._model_vars.vb = ca.SX.sym('vb', 3)
        self._model_vars.wb = ca.SX.sym('wb', 3)
        self._model_vars.u = ca.SX.sym('u', 4)
        self._model_vars.r = self.rot.r()

    def _compute_pose_evolution(self):

        vb = self._model_vars.vb
        wb = self._model_vars.wb
        R = self.rot.R()

        vg = R @ vb

        self._model_vars.p_dot = vg
        self._model_vars.r_dot = self.rot.M() @ wb

        self._model_vars.R = R
        self._model_vars.vg = vg

    def _compute_forces(self):
        R = self._model_vars.R
        e1 = R[:,0]
        e2 = R[:,1]
        e3 = R[:,2]
        vb = self._model_vars.vb
        wb = self._model_vars.wb

        Fgb = -self.config.m * self.config.g * ca.vertcat(e1[2], e2[2], e3[2])
        Fdb = [-self.config.b1, -self.config.b2, -self.config.b3] * vb
        Kdb = [-self.config.bw1, -self.config.bw2, -self.config.bw3] * wb

        u = self._model_vars.u
        Fa1 = 0
        Fa2 = 0
        Fa3 =   u[0] + u[1] + u[2] + u[3]
        Tb = ca.vertcat(Fa1, Fa2, Fa3)

        Ka1 = ( u[0] + u[1] - u[2] - u[3]) * self.config.l
        Ka2 = (-u[0] + u[1] + u[2] - u[3]) * self.config.l
        Ka3 = ( u[0] - u[1] + u[2] - u[3]) * self.config.k
        TKb = ca.vertcat(Ka1, Ka2, Ka3)

        Fb = Fdb + Fgb + Tb
        Kb = Kdb + TKb
        Tg = R @ Tb

        self._model_vars.Fb = Fb
        self._model_vars.Fgb = Fgb
        self._model_vars.Tb = Tb
        self._model_vars.Tg = Tg
        self._model_vars.Kb = Kb

    def _compute_state_evolution(self):
        p = self._model_vars.p
        r = self._model_vars.r
        vb = self._model_vars.vb
        wb = self._model_vars.wb

        p_dot = self._model_vars.p_dot
        r_dot = self._model_vars.r_dot

        Fb = self._model_vars.Fb
        Kb = self._model_vars.Kb
        def _hat(vec):
            return ca.vertcat(
                ca.horzcat(0, -vec[2], vec[1]),
                ca.horzcat(vec[2], 0, -vec[0]),
                ca.horzcat(-vec[1], vec[0], 0)
            )
        Wb = _hat(wb)
        Ib = np.diag([self.config.I1, self.config.I2, self.config.I3])

        vb_dot = Fb / self.config.m - Wb @ vb
        wb_dot = ca.inv(Ib) @ (Kb - Wb @ Ib @ wb)

        z = ca.vertcat(p, r, vb, wb)
        z_dot = ca.vertcat(p_dot, r_dot, vb_dot, wb_dot)

        self._model_vars.z = z
        self._model_vars.z_dot = z_dot
        self._model_vars.vb_dot = vb_dot
        self._model_vars.wb_dot = wb_dot

    def get_empty_state(self) -> DroneState:
        if self.config.global_r:
            if self.config.use_quat:
                r = GlobalQuaternion()
            else:
                r = GlobalEulerAngles()
        else:
            if self.config.use_quat:
                r = RelativeQuaternion()
            else:
                r = RelativeEulerAngles()
        return DroneState(r = r)

    def state2u(self, state:DroneState):
        ''' vectorize model input '''
        return state.u.to_vec()

    def state2zu(self, state:DroneState):
        ''' vectorize model state and input '''
        u = self.state2u(state)

        z = [
            *state.x.to_vec(),
            *state.r.to_vec(),
            *state.v.to_vec(),
            *state.w.to_vec()
        ]
        return z, u

    def u2state(self, state:DroneState, u):
        ''' update state with vectorized input '''
        state.u.from_vec(u)

    def du2state(self, state:DroneState, du):
        ''' update state with vectorized input '''
        state.du.from_vec(du)

    def zu2state(self, state:DroneState, z, u):
        ''' update state with vectorized state and input '''
        self.u2state(state, u)

        state.x.from_vec(z[:3])
        if self.config.use_quat:
            state.r.from_vec(z[3:7])
            state.v.from_vec(z[7:10])
            state.w.from_vec(z[10:13])

            state.q.from_vec(z[3:7])
        else:
            state.r.from_vec(z[3:6])
            state.v.from_vec(z[6:9])
            state.w.from_vec(z[9:12])

            R = self.f_R(z, u)
            state.q.from_mat(R)
            if self._last_q is not None:
                if np.linalg.norm(self._last_q - state.q.to_vec()) > 1.8:
                    state.q.from_vec(-state.q.to_vec())
            self._last_q = state.q.to_vec()

    def zu(self):
        ''' upper bound on state vector '''
        return [
            np.inf,
            np.inf,
            np.inf,
            *self.rot.ubr(),
            np.inf,
            np.inf,
            np.inf,
            self.config.w_max,
            self.config.w_max,
            self.config.w_max,
        ]

    def zl(self):
        ''' lower bound on state vector '''
        return[
            -np.inf,
            -np.inf,
            -np.inf,
            *self.rot.lbr(),
            -np.inf,
            -np.inf,
            -np.inf,
            self.config.w_min,
            self.config.w_min,
            self.config.w_min,
        ]

    def uu(self):
        ''' upper bound on input vector '''
        return [self.config.T_max] * self._model_vars.u.shape[0]

    def ul(self):
        ''' lower bound on input vector '''
        return [self.config.T_min] * self._model_vars.u.shape[0]

    def duu(self):
        ''' upper bound on input rate vector '''
        return [self.config.dT_max] * self._model_vars.u.shape[0]

    def dul(self):
        ''' lower bound on input rate vector '''
        return [self.config.dT_min] * self._model_vars.u.shape[0]

    def add_model_stage_constraints(self, z, u, g, lbg, ubg):
        # pylint: disable=unused-argument
        ''' currently no model stage constraints for drone models '''


class ParametricDroneModel(ParametricDynamicsModel, DroneModel):
    ''' drone model in arbitrary frame '''

    def __init__(self,
            config: DroneConfig,
            line: BaseCenterline):
        self.line = line
        super().__init__(config)

    def _create_vars(self):
        super()._create_vars()
        self._model_vars.p = self.line.sym_rep.p

    def _compute_pose_evolution(self):
        sym_rep = self.line.sym_rep

        vb = self._model_vars.vb
        wb = self._model_vars.wb
        Rp = sym_rep.Rp

        if self.config.global_r:
            R_rel = Rp.T @ self.rot.R()
        else:
            R_rel = self.rot.R()

        vp = R_rel @ vb

        y = sym_rep.p[1]
        n = sym_rep.p[2]
        s_dot = vp[0] / sym_rep.mag_xcs / (1 + sym_rep.ky*n - sym_rep.kn * y)
        y_dot = vp[1] + n * sym_rep.ks * s_dot * sym_rep.mag_xcs
        n_dot = vp[2] - y * sym_rep.ks * s_dot * sym_rep.mag_xcs
        p_dot = ca.vertcat(s_dot, y_dot, n_dot)

        wp = sym_rep.k * s_dot * sym_rep.mag_xcs
        if self.config.global_r:
            w_eff = wb
        else:
            w_eff = wb - self.rot.R().T @ wp

        r_dot = self.rot.M() @ w_eff

        self._model_vars.wp = wp
        self._model_vars.w_eff = w_eff
        self._model_vars.p_dot = p_dot
        self._model_vars.r_dot = r_dot

        # global rotation matrix from relative one
        if self.config.global_r:
            R = self.rot.R()
        else:
            R = Rp @ self.rot.R()
        self._model_vars.R = R
        self._model_vars.R_rel = R_rel

        vg = R @ vb
        self._model_vars.vg = vg

    def state2zu(self, state:DroneState):
        ''' vectorize model state and input '''
        u = self.state2u(state)

        z = [
            *state.p.to_vec(),
            *state.r.to_vec(),
            *state.v.to_vec(),
            *state.w.to_vec()
        ]
        return z, u

    def zu2state(self, state:DroneState, z, u):
        ''' update state with vectorized state and input '''
        self.u2state(state, u)

        state.p.from_vec(z[:3])
        if self.config.use_quat:
            state.r.from_vec(z[3:7])
            state.v.from_vec(z[7:10])
            state.w.from_vec(z[10:13])
        else:
            state.r.from_vec(z[3:6])
            state.v.from_vec(z[6:9])
            state.w.from_vec(z[9:12])

        state.x.from_vec(
            self.line.p2x(*state.p.to_vec())
        )
        R = self.f_R(z, u)
        state.q.from_mat(R)
        if self._last_q is not None:
            if np.linalg.norm(self._last_q - state.q.to_vec()) > 1.8:
                state.q.from_vec(-state.q.to_vec())
        self._last_q = state.q.to_vec()
