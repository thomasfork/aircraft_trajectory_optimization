''' dynamics model of a point mass '''
from typing import Callable

import casadi as ca
import numpy as np

from drone3d.pytypes import PointConfig, PointState
from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.dynamics.dynamics_model import DynamicsModel, \
    ParametricDynamicsModel, DynamicsModelVars


class PointModel(DynamicsModel):
    ''' point mass model in inertial frame '''
    config: PointConfig

    def __init__(self,
            config: PointConfig):
        self.config = config
        self._model_vars = DynamicsModelVars()
        self._setup()

    def _create_vars(self):
        self._model_vars.p = ca.SX.sym('x', 3)
        self._model_vars.vb = ca.SX.sym('vb', 3)
        self._model_vars.u = ca.SX.sym('u', 3)

    def _compute_pose_evolution(self):

        vb = self._model_vars.vb

        wb = np.array([0., 0., 0.])
        self._model_vars.wb = wb
        self._model_vars.p_dot = vb

        # global rotation matrix
        R = np.eye(3)
        self._model_vars.R = R

        self._model_vars.vg = vb

    def _compute_forces(self):
        R = self._model_vars.R
        e1 = R[:,0]
        e2 = R[:,1]
        e3 = R[:,2]
        vb = self._model_vars.vb

        Tb = self._model_vars.u
        Fgb = -self.config.m * self.config.g * ca.vertcat(e1[2], e2[2], e3[2])
        Fdb = [-self.config.b1, -self.config.b2, -self.config.b3] * vb

        Fb = Tb + Fgb + Fdb # body frame force
        Tg = R @ Tb         # global frame thrust

        self._model_vars.Fb = Fb
        self._model_vars.Fgb = Fgb
        self._model_vars.Tb = Tb
        self._model_vars.Tg = Tg

    def _compute_state_evolution(self):
        p = self._model_vars.p
        vb = self._model_vars.vb

        p_dot = self._model_vars.p_dot

        Fb = self._model_vars.Fb
        vb_dot = Fb / self.config.m

        z = ca.vertcat(p, vb)
        z_dot = ca.vertcat(p_dot, vb_dot)

        self._model_vars.z = z
        self._model_vars.z_dot = z_dot
        self._model_vars.vb_dot = vb_dot

    def get_empty_state(self) -> PointState:
        return PointState()

    def state2u(self, state:PointState):
        return state.u.to_vec()

    def state2zu(self, state:PointState):
        u = self.state2u(state)

        z = [
            *state.x.to_vec(),
            *state.v.to_vec()
        ]
        return z, u

    def u2state(self, state:PointState, u):
        state.u.from_vec(u)

    def du2state(self, state:PointState, du):
        state.du.from_vec(du)

    def zu2state(self, state:PointState, z, u):
        self.u2state(state, u)

        state.x.from_vec(z[:3])
        state.v.from_vec(z[3:6])

    def zu(self):
        return [np.inf] * self._model_vars.z.shape[0]

    def zl(self):
        return [-np.inf] * self._model_vars.z.shape[0]

    def uu(self):
        return [self.config.T_max] * self._model_vars.u.shape[0]

    def ul(self):
        return [self.config.T_min] * self._model_vars.u.shape[0]

    def duu(self):
        return [self.config.dT_max] * self._model_vars.u.shape[0]

    def dul(self):
        return [self.config.dT_min] * self._model_vars.u.shape[0]

    def add_model_stage_constraints(self, z, u, g, lbg, ubg):
        # pylint: disable=unused-argument
        ''' spherical constraint on input '''
        u_mag = u.T @ u

        g += [u_mag / self.config.T_max / self.config.T_max]
        ubg += [1]
        lbg += [-np.inf]

class ParametricPointModel(ParametricDynamicsModel, PointModel):
    ''' point mass model in parametric frame '''

    # function to get angular velocity
    # only nonzero if relative orientation
    f_w: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(self,
            config: PointConfig,
            line: BaseCenterline):
        self.line = line
        super().__init__(config)

    def _create_vars(self):
        self._model_vars.p = self.line.sym_rep.p
        self._model_vars.vb = ca.SX.sym('vb', 3)
        self._model_vars.u = ca.SX.sym('u', 3)

    def _compute_pose_evolution(self):
        sym_rep = self.line.sym_rep

        vb = self._model_vars.vb
        Rp = sym_rep.Rp

        if self.config.global_r:
            R_rel = Rp.T
        else:
            R_rel = np.eye(3)

        vp = R_rel @ vb

        y = sym_rep.p[1]
        n = sym_rep.p[2]
        s_dot = vp[0] / sym_rep.mag_xcs / (1 + sym_rep.ky*n - sym_rep.kn * y)
        y_dot = vp[1] + n * sym_rep.ks * s_dot * sym_rep.mag_xcs
        n_dot = vp[2] - y * sym_rep.ks * s_dot * sym_rep.mag_xcs
        p_dot = ca.vertcat(s_dot, y_dot, n_dot)

        wp = sym_rep.k * s_dot * sym_rep.mag_xcs
        if self.config.global_r:
            wb = np.array([0., 0., 0.])
        else:
            wb = wp

        self._model_vars.wb = wb
        self._model_vars.wp = wp
        self._model_vars.p_dot = p_dot

        # global rotation matrix from relative one
        if self.config.global_r:
            R = np.eye(3)
        else:
            R = Rp
        self._model_vars.R = R
        self._model_vars.R_rel = R_rel

        vg = R @ vb
        self._model_vars.vg = vg

    def _compute_state_evolution(self):
        p = self._model_vars.p
        vb = self._model_vars.vb
        wb = self._model_vars.wb

        p_dot = self._model_vars.p_dot

        Fb = self._model_vars.Fb

        def _hat(vec):
            return ca.vertcat(
                ca.horzcat(0, -vec[2], vec[1]),
                ca.horzcat(vec[2], 0, -vec[0]),
                ca.horzcat(-vec[1], vec[0], 0)
            )

        vb_dot = Fb / self.config.m - _hat(wb) @ vb

        z = ca.vertcat(p, vb)
        z_dot = ca.vertcat(p_dot, vb_dot)

        self._model_vars.z = z
        self._model_vars.z_dot = z_dot
        self._model_vars.vb_dot = vb_dot

    def _setup_helper_functions(self):
        super()._setup_helper_functions()

        z = self._model_vars.z
        u = self._model_vars.u
        param_terms = self.line.sym_rep.param_terms
        helper_args = [z,u, param_terms]

        # angular velocity induced by parametric frame
        w = self._model_vars.wb
        _, self.f_w = self._create_helper_function(
            'w',
            helper_args,
            [w])

    def state2zu(self, state:PointState):
        u = self.state2u(state)

        z = [
            *state.p.to_vec(),
            *state.v.to_vec()
        ]
        return z, u

    def zu2state(self, state:PointState, z, u):
        self.u2state(state, u)

        state.p.from_vec(z[:3])
        state.v.from_vec(z[3:6])

        state.x.from_vec(
            self.line.p2x(*state.p.to_vec())
        )
        #state.w.from_vec(
        #    self.f_w(z, u)
        #)
        R = self.f_R(z, u)
        state.q.from_mat(R)
