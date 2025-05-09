''' utilities for loading assets '''
from typing import Tuple

import numpy as np
import casadi as ca
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

from drone3d.utils.interp import linear_interpolant
from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.dynamics.dynamics_model import InterpolatedDynamicsModel
from drone3d.raceline.base_raceline import RacelineResults

def package_cpc_data_as_raceline(file: str, line: BaseCenterline, clip: bool = True) -> \
        Tuple[RacelineResults, InterpolatedDynamicsModel]:
    '''
    take raw data and package in a format suitable for plotting
    also returns a model for use in plotting
    '''
    model = InterpolatedDynamicsModel()

    data = np.genfromtxt(file, delimiter=',')[1:]
    t = data[:,0]
    x = data[:,1:4]
    q = data[:,[5,6,7,4]]
    v = data[:,8:11]
    w = data[:,11:14]

    # try to extract a single lap from cpc data
    if clip:
        x0 = line.p2xc(line.s_min())
        dist = cdist(x, x0[np.newaxis,:])
        i0 = dist.argmin()

        # estimate a lap time by interpolating position and finding closest to x[i0]
        x_interp = interp1d(t, x.T)
        def dist_from_prev_lap(t_guess: float):
            return np.linalg.norm(
                x[i0] - \
                x_interp(t_guess)
            )
        tp = np.linspace(t[i0+10], t.max(),1000)
        tf = tp[np.array([dist_from_prev_lap(t) for t in tp]).argmin()]
        lap_time = tf - t[i0]

        # final index to clip so that interpolation works
        i1 = np.searchsorted(t, tf) + 1

        t = t[i0:i1] - t[i0]
        x = x[i0:i1]
        q = q[i0:i1]
        v = v[i0:i1]
        w = w[i0:i1]
    else:
        lap_time = t[-1]

    # interpolate the data
    t_interp = ca.SX.sym('t')

    def interp_data(d):
        interps = [linear_interpolant(t, dk) for dk in d.T]
        return ca.Function(
            'interp',
            [t_interp],
            [ca.vertcat(*[x_i(t_interp) for x_i in interps])])

    z = np.hstack([x,q])
    u = np.hstack([v, w])

    states = []
    for (tk, zk, uk) in zip(t, z, u):
        state = model.get_empty_state()
        model.zu2state(state, zk, uk)
        state.t = tk
        states.append(state)

    z_interp = interp_data(z)
    u_interp = interp_data(u)
    du_interp = ca.Function('du', [t_interp], [ca.jacobian(u_interp(t_interp), t_interp)])

    def _z_interp_np(t):
        return np.array(z_interp(t)).squeeze()
    def _u_interp_np(t):
        return np.array(u_interp(t)).squeeze()
    def _du_interp_np(t):
        return np.array(du_interp(t)).squeeze()

    return RacelineResults(
        solve_time=-1,
        ipopt_time=-1,
        feval_time=-1,
        feasible=True,
        states = states,
        time = lap_time,
        label = 'CPC Data',
        color = [0.5,0,0.8,1],
        z_interp=_z_interp_np,
        u_interp=_u_interp_np,
        du_interp=_du_interp_np,
        global_frame=True
    ), model
