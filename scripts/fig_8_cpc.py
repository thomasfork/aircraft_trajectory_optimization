''' racelines through a loop to demonstrate warmstart and convexity '''
import numpy as np

from drone3d.centerlines.base_centerline import GateShape
from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
from drone3d.visualization.drone_raceline_fig import DroneRacelineWindow
from drone3d.utils.load_utils import get_assets_file
from drone3d.utils.cpc_utils import  package_cpc_data_as_raceline

def _main():
    x = np.array([0, 5, 0, -5,   0,  5,  0, -5])
    y = np.array([0, 1, 2,   1,  0,  -1,  -2, -1])
    z = np.array([10, 5, 0, -5, -10, -5, 0, 5])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.gate_shape = GateShape.CIRCLE
    config.gate_ri = 0.6
    config.gate_ro = 0.8
    line = SplineCenterline(config)

    cpc_raceline, cpc_model = package_cpc_data_as_raceline(
        get_assets_file('cpc_warmstart_raceline.csv'),
        line,
        clip=False
    )

    results = [
               cpc_raceline,
               ]
    solvers = [
               None,
               ]
    models =[
             cpc_model,
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
