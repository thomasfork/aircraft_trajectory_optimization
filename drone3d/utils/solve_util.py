''' utility for solving generic raceline problems '''
from typing import Tuple

from drone3d.pytypes import DroneConfig, PointConfig
from drone3d.centerlines.spline_centerline import SplineCenterline
from drone3d.raceline.base_raceline import GlobalRacelineConfig, ParametricRacelineConfig, \
    BaseRaceline, RacelineResults
from drone3d.raceline.drone_raceline import GlobalDroneRaceline,ParametricDroneRaceline
from drone3d.raceline.point_raceline import GlobalPointRaceline, ParametricPointRaceline

def solve_util(
    line: SplineCenterline,
    global_frame:bool,
    drone:bool,
    use_quaternion: bool = False,
    global_r: bool = True,
    use_ws: bool = False,
    solve:bool = True,
    fix_gate_center: bool = False,
    verbose: bool = True,
    use_rk4: bool = False,
    N = 50,
    v0 = 1.0
    )-> Tuple[BaseRaceline, RacelineResults]:
    '''
    utility for solving with default configurations with few lines of code
    '''

    if global_frame:
        config = GlobalRacelineConfig(verbose = verbose, N = N, v0 = v0, use_rk4=use_rk4)
        config.closed = line.config.closed
        config.gate_xi = line.config.x[0]
        config.gate_xj = line.config.x[1]
        config.gate_xk = line.config.x[2]
        config.fix_gate_center = fix_gate_center
        if drone:
            drone_config = DroneConfig(global_r=True, use_quat=use_quaternion)

            solver = GlobalDroneRaceline(
                line,
                config,
                drone_config,
                generate_ws=use_ws
            )
        else:
            drone_config = PointConfig(global_r=True)

            solver = GlobalPointRaceline(
                line,
                config,
                drone_config
            )
    else:
        config = ParametricRacelineConfig(verbose = verbose, N = N, v0 = v0, use_rk4=use_rk4)
        config.closed = line.config.closed
        config.fixed_gates = line.config.s[:-1] if line.config.closed \
            else line.config.s
        config.fix_gate_center = fix_gate_center
        if drone:
            drone_config = DroneConfig(global_r=global_r, use_quat=use_quaternion)

            solver = ParametricDroneRaceline(
                line,
                config,
                drone_config,
                generate_ws=use_ws
            )
        else:
            drone_config = PointConfig(global_r=global_r)

            solver = ParametricPointRaceline(
                line,
                config,
                drone_config
            )

    if solve:
        raceline = solver.solve()
    else:
        raceline = solver.get_ws()

    return solver, raceline
