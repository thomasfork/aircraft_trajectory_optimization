'''
drone raceline
'''
from typing import List, Tuple, TypeVar
import time

import numpy as np
import casadi as ca

import scipy.spatial.transform as transform

from drone3d.pytypes import DroneConfig, PointConfig
from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.dynamics.dynamics_model import DynamicsModel
from drone3d.dynamics.drone_models import ParametricDroneModel, DroneModel
from drone3d.obstacles.mesh_obstacle import MeshObstacle, ObstacleFreeTube
from drone3d.raceline.base_raceline import BaseParametricRaceline, GlobalRacelineConfig, \
    ParametricRacelineConfig, BaseGlobalRaceline, RacelineResults, BaseRaceline, \
    BaseParametricObstacleRaceline
from drone3d.raceline.point_raceline import GlobalPointRaceline, ParametricPointRaceline, \
    ParametricObstaclePointRaceline

BaseRacelineType = TypeVar('BaseRacelineType')
class DroneRaceline(BaseRaceline):
    ''' general purpose additions for drone racelines '''
    model: DroneModel
    ws_solver: BaseRaceline
    # recall first and previous warmstart orientation to help with continuity
    _first_ws_r = None
    _last_ws_r = None

    def _gemerate_ws(self, solver_class: BaseRacelineType, solver_args)\
            -> Tuple[RacelineResults, BaseRacelineType]:
        print('Generating Warmstart... ')
        t0 = time.time()
        self.ws_solver = solver_class(*solver_args)
        ws_raceline = self.ws_solver.solve()
        tf = time.time()
        print(f'Done (Lap Time: {ws_raceline.time:0.2f}s) (Setup + Solve: {tf-t0:0.2f}s)')
        return ws_raceline, self.ws_solver

    def _state_continuity_operator(self, z):
        if self.model.config.use_quat:
            z[3:7] = z[3:7] / ca.norm_2(z[3:7])
        return z

    def _enforce_modified_loop_closure(self):
        ''' modified with wrap-around for quaternions and euler angles '''
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
        z_delta = zF - z0

        g += [uF - u0]
        ubg += [0.] * input_dim[0]
        lbg += [0.] * input_dim[0]

        # first add the regular constraints
        if self.model.config.use_quat:
            g += [z_delta[1:3]]
            g += [z_delta[7:]]
        else:
            g += [z_delta[1:3]]
            g += [z_delta[4:]]

        # modified orientation continuity constraints
        if self.model.config.use_quat:
            # either qf = q0 or qf = -q0, pick one depending on warmstart

            # if warmstarted, use warmstart mismatch for constraints
            if self._first_ws_r is not None and self._last_ws_r is not None:
                if np.linalg.norm(self._first_ws_r - self._last_ws_r) > 1:
                    # warmstart is flipped, so flip constraint
                    g += [zF[3:7] + z0[3:7]]
                else:
                    g += [zF[3:7] - z0[3:7]]
            else:
                g += [zF[3:7] - z0[3:7]]
        else:
            if self._first_ws_r is not None and self._last_ws_r is not None:
                wraps = np.round((self._last_ws_r - self._first_ws_r)[0] / 2 / np.pi)
                g += [z_delta[3] - 2*np.pi*wraps]
            else:
                g += [z_delta[3]]

        if isinstance(self.model, ParametricDroneModel):
            ubg += [0.] * (state_dim[0] - 1)
            lbg += [0.] * (state_dim[0] - 1)

        else:
            g += [z_delta[0]]
            ubg += [0.] * state_dim[0]
            lbg += [0.] * state_dim[0]

    def _enforce_loop_closure(self):
        # deferred to self._build_decision_vector so that warmstart may be used.
        return

    def _enforce_initial_constraints(self):
        ''' stationary upright orientation for non-closed racelines '''
        super()._enforce_initial_constraints()
        z = self.nlp['Z'][0,0]
        u = self.nlp['U'][0,0]
        R = self.model.ca_f_R(z,u)
        e3 = R[:,2]

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        g += [e3]
        lbg += [0,0,1]
        ubg += [0,0,1]

        g += [z[-3:]]
        ubg += [0.]*3
        lbg += [0.]*3

    def _enforce_terminal_constraints(self):
        ''' stationary upright orientation for non-closed racelines '''
        super()._enforce_terminal_constraints()
        z = self._zF()
        u = self._uF()
        R = self.model.ca_f_R(z,u)
        e3 = R[:,2]

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        g += [e3]
        lbg += [0., 0., 1]
        ubg += [0., 0., 1]

        g += [z[-3:]]
        ubg += [0.]*3
        lbg += [0.]*3

    def _build_decision_vector(self):
        w, w0, ubw, lbw = super()._build_decision_vector()

        if self.config.closed:
            self._enforce_modified_loop_closure()

        return w, w0, ubw, lbw

    def _guess_z(self, n, k):
        z = super()._guess_z(n,k)
        if self.model.config.use_quat:
            z = [*z[:3], 1,0,0,0, *z[3:6], 0,0,0]
        else:
            z = [*z[:3], 0,0,0, *z[3:6], 0,0,0]

        if isinstance(self, BaseParametricRaceline):
            z[0] = self._get_s(n,k)

        # if using a warmstart, warmstart orientation
        if self._ws_available():
            # find where to interpolate
            idx = n * (self.config.K+1) + k
            t = self.ws_raceline.states[idx].t
            z_ws = self.ws_raceline.z_interp(t)
            u_ws = self.ws_raceline.u_interp(t)

            # warmstart position
            z[0] = z_ws[0]
            z[1] = z_ws[1]
            z[2] = z_ws[2]

            T = self.ws_model.f_T(z_ws, u_ws)

            if self.config.closed:
                # orientation matrix warmstart from thrust and velocity
                e1 = self.ws_model.f_vg(z_ws, u_ws)
                e1 = e1 / np.linalg.norm(e1)
                e3 = T / np.linalg.norm(T)
                e1 = e1 - e3 * (e1.T @ e3)
                e1 = e1 / np.linalg.norm(e1)
                e2 = np.cross(e3, e1)
                R = np.array([e1, e2, e3]).T
            else:
                # orientation matrix warmstart from thrust alone
                def _hat(v):
                    return np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                b = np.array([0, 0, 1])
                v =-np.cross(T / np.linalg.norm(T), b)
                s = np.linalg.norm(v)
                c = np.dot(T / np.linalg.norm(T), b)
                R = np.eye(3) + _hat(v) + _hat(v) @ _hat(v) *(1-c) / s**2

            if not self.model.config.global_r:
                Rp = self._guess_Rp(n, k)
                R = Rp.T @ R

            # apply orientation warmstart
            if self.model.config.use_quat:
                r = transform.Rotation.from_matrix(R).as_quat()
                if self._last_ws_r is not None:
                    delta = np.linalg.norm(r - self._last_ws_r)
                    # flip quaternion sign if it is discontinuous from previous one
                    if delta >= 1:
                        r = -r
                z[3] = r[0]
                z[4] = r[1]
                z[5] = r[2]
                z[6] = r[3]
            else:
                r = np.flip(transform.Rotation.from_matrix(R).as_euler('xyz', degrees=False))
                if self._last_ws_r is not None:
                    delta = np.linalg.norm(r - self._last_ws_r)
                    if delta > 1:
                        if r[0] - self._last_ws_r[0] > np.pi:
                            r[0] -= 2*np.pi
                        elif r[0] - self._last_ws_r[0] <= -np.pi:
                            r[0] += 2*np.pi

                        delta = np.linalg.norm(r - self._last_ws_r)
                        if delta > 1:
                            raise NotImplementedError(
                            'Warmstart continuity failed for euler angles, try quaternion')
                z[3] = r[0]
                z[4] = r[1]
                z[5] = r[2]

            # store orientation warmstart for continuity checks
            if self._first_ws_r is None:
                self._first_ws_r = r
            self._last_ws_r = r

            # now warmstart linear velocity
            vb = R.T @ self.ws_model.f_vg(z_ws, u_ws)
            z[-6] = vb[0]
            z[-5] = vb[1] # should be 0 from how R warmstart is built
            z[-4] = vb[2]

            # and warmstart angular velocity (from thrust derivative)
            du_ws = self.ws_raceline.du_interp(t)

            # spurious type hint for https://github.com/microsoft/pylance-release/issues/3277
            dT: float = self.ws_model.f_T(z_ws, du_ws)
            wg = np.cross(T, dT)
            wb = R.T @ wg / np.linalg.norm(T)**2
            z[-3] = wb[0]
            z[-2] = wb[1]
            z[-1] = wb[2]

        return z

    def _guess_u(self, n, k):
        if self._ws_available():
            # find where to interpolate
            idx = n * (self.config.K+1) + k
            t = self.ws_raceline.states[idx].t
            z_ws = self.ws_raceline.z_interp(t)
            u_ws = self.ws_raceline.u_interp(t)
            T = self.ws_model.f_T(z_ws, u_ws)
            return [np.linalg.norm(T)/4] * 4
        else:
            return super()._guess_u(n,k)

    def _guess_Rp(self, n, k):
        raise NotImplementedError('')


class GlobalDroneRaceline(DroneRaceline, BaseGlobalRaceline):
    ''' global drone raceline '''
    model: DroneModel
    ws_solver: GlobalPointRaceline

    def __init__(self,
            line: BaseCenterline,
            config: GlobalRacelineConfig,
            vehicle_config: DroneConfig,
            ws_raceline: RacelineResults = None,
            ws_model: DynamicsModel = None,
            generate_ws:bool = True):

        if generate_ws:
            ws_config = config.copy()
            ws_config.verbose = False
            ws_config.plot_iterations = False
            point_config = PointConfig(
                global_r=vehicle_config.global_r,
                collision_radius=vehicle_config.collision_radius,
            )
            ws_raceline, ws_solver = self._gemerate_ws(
                GlobalPointRaceline,
                (line, ws_config, point_config)
            )
            ws_model = ws_solver.model

        super().__init__(
            line = line,
            config = config,
            vehicle_config = vehicle_config,
            ws_raceline = ws_raceline,
            ws_model = ws_model)

    def _get_model(self, config: DroneConfig) -> DroneModel:
        config.global_r = True
        return DroneModel(config)

    def _get_prediction_color(self) -> List[float]:
        return [0,1,0,1]

    def _get_prediction_label(self) -> str:
        return 'Global Drone'


class ParametricDroneRaceline(DroneRaceline, BaseParametricRaceline):
    ''' parametric drone raceline '''
    model: ParametricDroneModel
    ws_solver: ParametricPointRaceline

    def __init__(self,
            line: BaseCenterline,
            config: ParametricRacelineConfig,
            vehicle_config: DroneConfig,
            ws_raceline: RacelineResults = None,
            ws_model: DynamicsModel = None,
            generate_ws:bool = True):

        if generate_ws:
            ws_config = config.copy()
            ws_config.verbose = False
            ws_config.plot_iterations = False
            point_config = PointConfig(
                global_r=vehicle_config.global_r,
            )
            ws_raceline, ws_solver = self._gemerate_ws(
                ParametricPointRaceline,
                (line, ws_config, point_config)
            )
            ws_model = ws_solver.model

        super().__init__(line, config, vehicle_config, ws_raceline, ws_model)

    def _get_model(self, config: DroneConfig) -> ParametricDroneModel:
        return ParametricDroneModel(
            config,
            self.line
        )

    def _get_prediction_color(self) -> List[float]:
        return [1,0,0,1]

    def _get_prediction_label(self) -> str:
        return 'Parametric Drone'

    def _guess_Rp(self, n, k):
        s = self._get_s(n,k)
        Rp = self.line.p2Rp(s)
        return Rp


class ParametricObstacleDroneRaceline(DroneRaceline, BaseParametricObstacleRaceline):
    ''' parametric drone raceline with obstacle avoidance'''
    model: ParametricDroneModel
    ws_solver: ParametricObstaclePointRaceline

    def __init__(self,
            line: BaseCenterline,
            config: ParametricRacelineConfig,
            vehicle_config: DroneConfig,
            mesh_obstacle: MeshObstacle,
            tube: ObstacleFreeTube = None,
            ws_raceline: RacelineResults = None,
            ws_model: DynamicsModel = None,
            generate_ws:bool = True):

        if generate_ws:
            ws_config = config.copy()
            ws_config.verbose = False
            ws_config.plot_iterations = False
            point_config = PointConfig(
                global_r=vehicle_config.global_r,
                collision_radius=vehicle_config.collision_radius

            )
            ws_raceline, ws_solver = self._gemerate_ws(
                ParametricObstaclePointRaceline,
                (line, ws_config, point_config, mesh_obstacle)
            )
            ws_model = ws_solver.model
            tube = ws_solver.tube

        super().__init__(
            line = line,
            config = config,
            vehicle_config = vehicle_config,
            mesh_obstacle = mesh_obstacle,
            tube = tube,
            ws_raceline = ws_raceline,
            ws_model = ws_model
        )

    def _get_model(self, config: DroneConfig) -> ParametricDroneModel:
        return ParametricDroneModel(
            config,
            self.line
        )

    def _get_prediction_color(self) -> List[float]:
        return [0,.3,1,1]

    def _get_prediction_label(self) -> str:
        return 'Drone w/ obstacles'

    def _guess_Rp(self, n, k):
        s = self._get_s(n,k)
        Rp = self.line.p2Rp(s)
        return Rp


def _main_global():
    # pylint: disable=import-outside-toplevel
    from drone3d.centerlines.base_centerline import GateShape
    from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
    from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow
    x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8])
    y = np.array([-1.6, 6.6, -4, -6, -6, -0.9, 6.8])
    z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.gate_shape = GateShape.CIRCLE
    line = SplineCenterline(config)

    config = GlobalRacelineConfig(verbose = True, N = len(x)*10)
    config.closed = line.config.closed
    config.gate_xi = x
    config.gate_xj = y
    config.gate_xk = z
    config.fix_gate_center = False
    config.plot_iterations = False
    config.use_rk4 = True
    drone_config = DroneConfig(global_r=True, use_quat=False)

    solver = GlobalDroneRaceline(
        line,
        config,
        drone_config,
        generate_ws=True,
    )

    raceline = solver.solve()
    DroneRacelineWindow(
        line,
        solver.model,
        raceline
    )

def _main():
    # pylint: disable=import-outside-toplevel
    from drone3d.centerlines.base_centerline import GateShape
    from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
    from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow
    x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8])
    y = np.array([-1.6, 6.6, -4, -6, -6, -0.9, 6.8])
    z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2])

    #x = np.array([0, 5, 0, -5,   0,  5,  0, -5])
    #y = np.array([0, 1, 2,   1,  0,  -1,  -2, -1])
    #z = np.array([10, 5, 0, -5, -10, -5, 0, 5])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.gate_shape = GateShape.SQUARE
    line = SplineCenterline(config)

    config = ParametricRacelineConfig(verbose = True, N = len(x)*10)
    config.closed = line.config.closed
    config.fixed_gates = line.config.s[:-1]
    drone_config = DroneConfig(global_r=True, use_quat=True)

    solver = ParametricDroneRaceline(
        line,
        config,
        drone_config,
        generate_ws=False
    )
    raceline = solver.solve()

    DroneRacelineWindow(
        line,
        solver.model,
        raceline,
    )

def _main_obstacles():
    # pylint: disable=import-outside-toplevel
    from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
    from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow

    x = np.array([-5,  -2.75, -0.66, 2.95, 8.67,  9.2, 1.57,-2.39, -4.7,-2.39,  4.23, -2.66])
    y = np.array([4.5, -0.08, -1.36, 1.25, 6.69, -3.6,-6.43, -6,   -6.43, -6.23, -0.66,  6.66])
    z = np.array([1.2,  2.815, 3.9, 2.815, 1.0,  1.0, 2.815, 3.9,  2.815, 1.0,   1.0,   1.0])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    line = SplineCenterline(config)
    if config.closed:
        line.config.gate_s = None
    else:
        line.config.gate_s = [line.s_min(), line.s_max()]

    mesh_env = MeshObstacle()

    config = ParametricRacelineConfig(verbose = True, N = 100)
    config.closed = line.config.closed
    drone_config = DroneConfig(global_r=True, use_quat=True, collision_radius=0.4)

    solver = ParametricObstacleDroneRaceline(
        line,
        config,
        drone_config,
        mesh_env,
        generate_ws=True
    )
    raceline = solver.solve()

    racelines = [raceline, solver.ws_raceline]
    models = [solver.model, solver.ws_model]

    window = DroneRacelineWindow(
        line,
        models,
        racelines,
        obstacles={'Environment':mesh_env},
        fullscreen=False,
        run = False
    )
    for name, obj in solver.triangulate_setup_info(window.ubo).items():
        window.add_object(name, obj, show = False)
    window.update_projection()
    window.run()

if __name__ == '__main__':
    _main_global()
    #_main()
    #_main_obstacles()
