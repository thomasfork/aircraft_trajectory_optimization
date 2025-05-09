''' racelines through a loop to demonstrate warmstart and nonconvexity '''
import numpy as np

from drone3d.centerlines.base_centerline import GateShape
from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow
from drone3d.utils.solve_util import solve_util

def _main():
    x = np.array([0, 5, 0, -5,   0,  5,  0, -5])
    y = np.array([0, 1, 2,   1,  0,  -1,  -2, -1])
    z = np.array([10, 5, 0, -5, -10, -5, 0, 5])

    N = 50

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.gate_shape = GateShape.CIRCLE
    line = SplineCenterline(config)

    baseline_solver, baseline_raceline = solve_util(
        line = line,
        global_frame=False,
        drone=True,
        use_quaternion=True,
        global_r=True,
        use_ws=False,
        N=N)
    baseline_raceline.label = 'Drone coldstart'
    baseline_raceline.color = [0,0,1,1]

    baseline_solver_euler, baseline_raceline_euler = solve_util(
        line = line,
        global_frame=False,
        drone=True,
        use_quaternion=False,
        global_r=True,
        use_ws=False,
        N=N)
    baseline_raceline_euler.label = 'Drone coldstart (Euler)'
    baseline_raceline_euler.color = [0,1,0,1]

    solver, raceline = solve_util(
        line = line,
        global_frame=False,
        drone=True,
        use_quaternion=True,
        global_r=True,
        use_ws=True,
        N=N)
    raceline.label = 'Drone with warmstart'
    raceline.color = [1,0,0,1]
    solver.ws_raceline.label = 'Point Mass WS'

    global_solver, global_raceline = solve_util(
        line = line,
        global_frame=True,
        drone=True,
        use_quaternion=True,
        global_r=True,
        use_ws=True,
        N=N)
    global_raceline.label = 'Global Drone with ws'

    results = [raceline, baseline_raceline, baseline_raceline_euler, global_raceline,
               solver.ws_raceline]
    solvers = [solver, baseline_solver, baseline_solver_euler, global_solver,
               solver.ws_solver]
    models =[solver.model, baseline_solver.model, baseline_solver_euler.model, global_solver.model,
             solver.ws_model]

    print('Raceline       Times: \tLap \tIPOPT \tNLP \tTotal \tSetup')
    delim = 's' # replace with ' & ' to make LaTex export easier
    for (solver, result) in zip(solvers, results):
        print(f'{result.label:20s}' +
              f'\t{result.time:0.3f}' + delim +
              f'\t{result.ipopt_time:0.3f}' + delim +
              f'\t{result.feval_time:0.3f}' + delim +
              f'\t{result.solve_time:0.3f}' + delim +
              (f'\t{solver.setup_time:0.3f}'+ delim if solver is not None else ''))

    DroneRacelineWindow(
        line,
        results=results,
        models = models
    )

if __name__ == '__main__':
    _main()
