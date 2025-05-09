'''
point mass raceline
'''
from typing import List

from drone3d.pytypes import PointConfig
from drone3d.dynamics.point_model import ParametricPointModel, PointModel
from drone3d.raceline.base_raceline import BaseParametricRaceline, ParametricRacelineConfig, \
    BaseGlobalRaceline, GlobalRacelineConfig, BaseRaceline, BaseParametricObstacleRaceline


class PointRaceline(BaseRaceline):
    ''' base point mass raceline '''

    def _enforce_initial_constraints(self):
        super()._enforce_initial_constraints()

        # constrain thust to be purely upwards, ie. to a drone pointing up
        z = self.nlp['Z'][0,0]
        u = self.nlp['U'][0,0]
        T = self.model.ca_f_T(z,u)

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        g += [T[0], T[1]]
        ubg += [0., 0.]
        lbg += [0., 0.]

    def _enforce_terminal_constraints(self):
        super()._enforce_terminal_constraints()

        # constrain thust to be purely upwards, ie. to a drone pointing up
        z = self._zF()
        u = self._uF()
        T = self.model.ca_f_T(z,u)

        g = self.nlp['g']
        ubg = self.nlp['ubg']
        lbg = self.nlp['lbg']

        g += [T[0], T[1]]
        ubg += [0., 0.]
        lbg += [0., 0.]


class GlobalPointRaceline(PointRaceline, BaseGlobalRaceline):
    ''' global point mass raceline '''

    def _get_model(self, config: PointConfig) -> PointModel:
        return PointModel(config)

    def _get_prediction_color(self) -> List[float]:
        return [1,1,1,1]

    def _get_prediction_label(self) -> str:
        return 'Global PM'


class ParametricPointRaceline(PointRaceline, BaseParametricRaceline):
    ''' parametric point mass raceline '''

    def _get_model(self, config: PointConfig) -> ParametricPointModel:
        return ParametricPointModel(
            config,
            self.line
        )

    def _get_prediction_color(self) -> List[float]:
        return [.5,.5,.5,1]

    def _get_prediction_label(self) -> str:
        return 'Parametric PM'


class ParametricObstaclePointRaceline(PointRaceline, BaseParametricObstacleRaceline):
    ''' parametric point mass raceline with obstacle avoidance'''

    def _get_model(self, config: PointConfig) -> ParametricPointModel:
        return ParametricPointModel(
            config,
            self.line
        )

    def _get_prediction_color(self) -> List[float]:
        return [.3,.3,.3,1]

    def _get_prediction_label(self) -> str:
        return 'PM w/ obstacles'


def _main_global():
    # pylint: disable=import-outside-toplevel
    from drone3d.centerlines.base_centerline import GateShape
    from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
    from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow
    import numpy as np
    x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8])
    y = np.array([-1.6, 6.6, -4, -6, -6, -0.9, 6.8])
    z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.gate_shape = GateShape.SQUARE
    line = SplineCenterline(config)

    config = GlobalRacelineConfig(verbose = True, N = 50)
    config.closed = line.config.closed
    config.gate_xi = x
    config.gate_xj = y
    config.gate_xk = z
    config.fix_gate_center = False
    config.use_rk4 = True
    drone_config = PointConfig(global_r=True)

    solver = GlobalPointRaceline(
        line,
        config,
        drone_config
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
    import numpy as np
    x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8])
    y = np.array([-1.6, 6.6, -4, -6, -6, -0.9, 6.8])
    z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.gate_shape = GateShape.SQUARE
    line = SplineCenterline(config)

    config = ParametricRacelineConfig(verbose = True, N = 50)
    config.closed = line.config.closed
    config.use_rk4 = True
    drone_config = PointConfig(global_r=True)

    solver = ParametricPointRaceline(
        line,
        config,
        drone_config
    )

    raceline = solver.solve()
    DroneRacelineWindow(
        line,
        solver.model,
        raceline
    )

def _main_obstacles():
    # pylint: disable=import-outside-toplevel
    from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
    from drone3d.obstacles.mesh_obstacle import MeshObstacle
    from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow
    import numpy as np

    x = np.array([-5,  -2.75, -0.66, 2.95, 8.67,  9.2, 1.57,-2.39, -4.7,-2.39,  4.23, -2.66, -6.0])
    y = np.array([4.5, -0.08, -1.36, 1.25, 6.69, -3.6,-6.43, -6,   -6.43, -6.23, -0.66,  6.66,  0])
    z = np.array([1.2,  2.815, 3.9, 2.815, 1.0,  1.0, 2.815, 3.9,  2.815, 1.0,   1.0,   1.0,   1.0])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = False
    line = SplineCenterline(config)
    line.config.gate_s = [line.s_min(), line.s_max()]

    mesh_env = MeshObstacle()

    config = ParametricRacelineConfig(verbose = True, N = 100)
    config.closed = line.config.closed
    drone_config = PointConfig(global_r=True, collision_radius=0.4)

    solver = ParametricObstaclePointRaceline(
        line,
        config,
        drone_config,
        mesh_env
    )
    raceline = solver.solve()

    print(solver.calc_tube_time)

    window = DroneRacelineWindow(
        line,
        solver.model,
        raceline,
        obstacles={'Environment':mesh_env},
        run = False
    )
    for name, obj in solver.triangulate_setup_info(window.ubo).items():
        window.add_object(name, obj, show = False)
    window.update_projection()
    window.run()

if __name__ == '__main__':
    #_main_global()
    _main()
    #_main_obstacles()
