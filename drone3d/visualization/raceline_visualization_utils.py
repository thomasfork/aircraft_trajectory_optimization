'''
visualization utilities
for OpenGL and imgui
specifically for racelines, separate to avoid circur dependencies
'''

from typing import Tuple

import numpy as np

from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.raceline.base_raceline import RacelineResults
from drone3d.visualization.shaders import vtype
from drone3d.visualization.utils import get_cmap_rgba

def load_trajectory(
        line: BaseCenterline,
        plan: RacelineResults,
        n: int = 1000,
        w: float = 0.05,
        h: float = 0.05,
        v_max: float = None,
        v_min: float = None,
        closed: bool = None,
        global_frame: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
    '''
    turn a trajectory, a list of vehicle states,
    into vertex/index data for plotting
    '''
    # generate state data
    t = np.linspace(0, plan.time, n)
    z = plan.z_interp(t[None])
    if z.shape[0] == 6: # point mass
        v = np.linalg.norm(z[3:], axis = 0)
    else:
        v = np.linalg.norm(z[-6:-3], axis = 0)

    # check optional arguments
    v_max = v_max if v_max is not None else v.max()
    v_min = v_min if v_min is not None else v.min()
    if v_max == v_min:
        v_max += 1e-6
    closed = closed if closed is not None else line.config.closed

    # get position and normal/lateral vectors
    if global_frame:
        x = z[0:3].T
        dx = np.gradient(x, axis = 0)
        ex = dx / np.linalg.norm(dx, axis = 1)[:,np.newaxis]
        e2 = np.zeros(x.shape)
        e3 = np.zeros(x.shape)
        e2[:,0] = ex[:,1]
        e2[:,1] = -ex[:,0]
        e3[:,2] = 1
    else:
        x = line.p2x(z[0:1], z[1:2], z[2:3]).T
        e2 = line.p2ey(z[0:1]).T
        e3 = line.p2en(z[0:1]).T

    V = np.concatenate([x + e2*w + e3*h,
                        x + e2*w,
                        x - e2*w,
                        x - e2*w + e3*h])

    C = get_cmap_rgba(v, v_min, v_max)
    C = np.tile(C.T, 4).T

    N = np.zeros(V.shape)
    N[:,-1] = 1

    # first face
    I = np.array([[0,1,n],
                  [1,n+1,n]])
    # first segment
    I = np.concatenate([I, I+n, I+2*n, I+3*n])
    I = I % (n*4)
    #all segments
    I = np.concatenate([I + k for k in range(n-1)])

    if not closed:
        # cap the ends
        I = np.concatenate([I, np.array([[0,n,2*n],
                                         [0,2*n, 3*n]])])
        I = np.concatenate([I, n-1+ np.array([[0,2*n,n],
                                     [0,3*n, 2*n]])])
    else:
        # join start and end
        I_join =  np.array([[0,n,n-1], [n,2*n-1,n-1]])
        I_join = np.concatenate([I_join, I_join+n, I_join+2*n, I_join+3*n]) % (4*n)

        I = np.concatenate([I, I_join])

    Vertices = np.zeros(V.shape[0], dtype=vtype)
    Vertices['a_position'] = V
    Vertices['a_color']    = C
    Vertices['a_normal']   = N

    return Vertices, np.concatenate(I).astype(np.uint32), v_max, v_min
