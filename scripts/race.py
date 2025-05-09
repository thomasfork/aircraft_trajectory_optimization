''' point mass racelines '''
import numpy as np

from drone3d.centerlines.base_centerline import GateShape
from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow
from drone3d.utils.load_utils import get_assets_file
from drone3d.utils.cpc_utils import  package_cpc_data_as_raceline
from drone3d.utils.solve_util import solve_util

def _main():
    x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8])
    y = np.array([-1.6, 6.6, -4, -6, -6, -0.9, 6.8])
    z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    # shrink gate size to replicate distance tolerance of 0.3m of CPC
    # (0.3m is removed from collision radius 'ri' and 'ro' is cosmetic outer radius)
    #config.gate_ri = 0.6
    #config.gate_ro = 0.7
    #config.gate_shape = GateShape.CIRCLE

    # full gate shape
    config.gate_shape = GateShape.SQUARE

    line = SplineCenterline(config)

    N = len(x) * 10
    use_rk4: bool = True
    use_quaternion: bool = True

    global_solver, global_raceline = solve_util(
        line = line,
        global_frame=True,
        drone=True,
        use_ws=True,
        use_quaternion=use_quaternion,
        use_rk4 = use_rk4,
        N = N)

    parametric_solver, parametric_raceline = solve_util(
        line = line,
        global_frame=False,
        drone=True,
        use_ws=True,
        use_quaternion=use_quaternion,
        use_rk4 = use_rk4,
        N = N)

    cpc_raceline, cpc_model = package_cpc_data_as_raceline(
        get_assets_file('cpc_race_raceline.csv'),
        line
    )

    results = [
        global_raceline,
        parametric_raceline,
        cpc_raceline,
        global_solver.ws_raceline,
        parametric_solver.ws_raceline,
        ]
    solvers = [
        global_solver,
        parametric_solver,
        None,
        global_solver.ws_solver,
        parametric_solver.ws_solver
    ]
    models =[
        global_solver.model,
        parametric_solver.model,
        cpc_model,
        global_solver.ws_model,
        parametric_solver.ws_model,
        ]

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
