'''
main script for obstacle avoidance demo
'''
import numpy as np

from drone3d.pytypes import DroneConfig
from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
from drone3d.raceline.base_raceline import ParametricRacelineConfig
from drone3d.raceline.drone_raceline import ParametricObstacleDroneRaceline
from drone3d.obstacles.mesh_obstacle import MeshObstacle
from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow

def _main():
    x = np.array([-5,  -2.75, -0.66, 2.95, 8.67,  9.2, 1.57,-2.39, -4.7,-2.39,  4.23, -2.66])
    y = np.array([4.5, -0.08, -1.36, 1.25, 6.69, -3.6,-6.43, -6,   -6.43, -6.23, -0.66,  6.66])
    z = np.array([1.2,  2.815, 3.9, 2.815, 1.0,  1.0, 2.815, 3.9,  2.815, 1.0,   1.0,   1.0])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    line = SplineCenterline(config)
    # unecessary gates since tube does majority of constraints
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

    results = [raceline, solver.ws_raceline]
    models = [solver.model, solver.ws_model]
    solvers = [solver, solver.ws_solver]

    print(f'Tube Compute Time: {solver.ws_solver.calc_tube_time:0.3f}s')
    print('Raceline       Times: \tLap \tIPOPT \tNLP \tTotal \tSetup')
    delim = 's' # replace with ' & ' to make LaTex export easier
    for (solver, result) in zip(solvers, results):
        print(f'{result.label:20s}' +
              f'\t{result.time:0.3f}' + delim +
              f'\t{result.ipopt_time:0.3f}' + delim +
              f'\t{result.feval_time:0.3f}' + delim +
              f'\t{result.solve_time:0.3f}' + delim +
              (f'\t{solver.setup_time:0.3f}'+ delim if solver is not None else ''))

    window = DroneRacelineWindow(
        line,
        models,
        results,
        obstacles={'Environment':mesh_env},
        fullscreen=False,
        run = False
    )
    for name, obj in solver.triangulate_setup_info(window.ubo).items():
        window.add_object(name, obj, show = False)
    window.update_projection()
    window.run()


if __name__ == '__main__':
    _main()
