'''
visualization utilities
for OpenGL and imgui
'''
from typing import List, Tuple
from operator import itemgetter

import numpy as np
import imgui

from drone3d.visualization.shaders import vtype

def get_cmap_rgba(v, v_min, v_max):
    ''' get RGBA color for the colormap to be used for vehicle speed on plots'''
    v_rel = (v - v_min) / (v_max - v_min)
    rgba = np.array([
        np.interp(v_rel, [0,0.5,1.0], [0,.8, 1]),
        np.interp(v_rel, [0,0.5,1.0], [0,.3, 1]),
        np.interp(v_rel, [0,0.5,1.0], [.5,.5,0]),
        np.ones(v_rel.shape)
    ]).T
    return rgba

def join_vis(vis):
    ''' join a list of tuples of vertex/index data '''
    Vertices = None
    Indices = None
    for (V, I) in vis:
        if Vertices is None or Indices is None:
            Vertices = V
            Indices = I
        else:
            Indices = np.concatenate([Indices, I + len(Vertices)])
            Vertices = np.concatenate([Vertices, V])

    return Vertices, Indices.astype(np.uint32)

def get_circular_gate(
        n: int = 500,
        ri: float = 1.25,
        ro: float = 1.35,
        w: float  = 0.2,
        color: List[float] = None)-> Tuple[np.ndarray, np.ndarray]:
    ''' internal helper function for making circular racetrack gates '''
    if color is None:
        color = [0, 1, 0, 1]
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    X_lower_inner = ri * np.array([np.zeros(n), np.cos(th), np.sin(th)]).T
    X_lower_outer = ro * np.array([np.zeros(n), np.cos(th), np.sin(th)]).T
    X_upper_inner = X_lower_inner.copy()
    X_upper_outer = X_lower_outer.copy()
    X_lower_inner[:,0] -= w/2
    X_lower_outer[:,0] -= w/2
    X_upper_inner[:,0] += w/2
    X_upper_outer[:,0] += w/2

    N_upper_outer = np.array([np.ones(n), np.cos(th), np.sin(th)]).T
    N_upper_outer = N_upper_outer / np.linalg.norm(X_lower_outer, axis = 1)[:, np.newaxis]
    N_upper_inner = -N_upper_outer.copy()
    N_upper_inner[:,0] *= -1
    N_lower_outer = N_upper_outer.copy()
    N_lower_outer[:,0] *= -1
    N_lower_inner = -N_lower_outer.copy()
    N_lower_inner[:,0] *= -1

    V = np.concatenate([
        X_lower_inner,
        X_lower_outer,
        X_upper_outer,
        X_upper_inner
    ])
    N = np.concatenate([
        N_lower_inner,
        N_lower_outer,
        N_upper_outer,
        N_upper_inner
    ])

    # first face
    n_wrap = n
    I = np.array([[0,1,n_wrap],
                [1, n_wrap+1, n_wrap]])

    # first segment
    I = np.concatenate([I+k for k in range(n)])
    I[-1,0] -= n
    I[-1,1] -= n
    I[-2,1] -= n

    # ring
    I = np.concatenate([I + k for k in range(0, 4*n, n)])
    I = np.mod(I, n*4)


    Vertices = np.zeros(V.shape[0], dtype=vtype)
    Vertices['a_position'] = V
    Vertices['a_color']    = color
    Vertices['a_normal']   = N

    return Vertices, np.concatenate(I).astype(np.uint32)

def get_square_gate(
        ri: float = 1.25,
        ro: float = 1.35,
        w: float = 0.2,
        color: List[float] = None)-> Tuple[np.ndarray, np.ndarray]:
    ''' internal helper function for making square racetrack gates '''
    if color is None:
        color = [0, 1, 0, 1]

    X = np.array([
        [0, ri, ri],
        [0, -ri, ri],
        [0, -ri, -ri],
        [0, ri, -ri]
    ])
    N = np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, 1, 1],
        [-1, -1, 1],
    ]) / np.sqrt(3)

    X2 = X.copy()
    X[:,0] -= w/2
    X2[:,0] += w/2

    N2 = N.copy()
    N2[:,0] *= -1

    Xo = np.array([
        [0, ro, ro],
        [0, -ro, ro],
        [0, -ro, -ro],
        [0, ro, -ro]
    ])
    No = -N2

    Xo2 = Xo.copy()
    Xo[:,0] -= w/2
    Xo2[:,0] += w/2

    No2 = No.copy()
    No2[:,0] *= -1

    X = np.vstack([X, X2, Xo2, Xo])
    N = np.vstack([N, N2, No2, No])

    Vertices = np.zeros(X.shape[0], dtype=vtype)
    Vertices['a_position'] = X
    Vertices['a_color']    = color
    Vertices['a_normal']   = N

    I = np.array([
        [0,4,1],
        [1,4,5]
        ])

    I = np.concatenate([I+k for k in range(4)])
    I[-2,2] -= 4
    I[-1,0] -= 4
    I[-1,2] -= 4

    I = np.concatenate([I+k*4 for k in range(4)])
    I = np.mod(I, 16)

    return Vertices, np.concatenate(I).astype(np.uint32)

def get_rectangular_gate(
        a: float = 2,
        b: float = 2,
        w: float = 0.2,
        color: List[float] = None):
    ''' rectangular gate with width a, length b, and thickness w '''
    if color is None:
        color = [0, 1, 0, 1]

    X = np.array([
        [0, a/2, b/2],
        [0, -a/2, b/2],
        [0, -a/2, -b/2],
        [0, a/2, -b/2]
    ])
    N = np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, 1, 1],
        [-1, -1, 1],
    ]) / np.sqrt(3)

    X2 = X.copy()
    X[:,0] -= w/2
    X2[:,0] += w/2

    N2 = N.copy()
    N2[:,0] *= -1

    Xo = np.array([
        [0, a/2+w, b/2+w],
        [0, -a/2-w, b/2+w],
        [0, -a/2-w, -b/2-w],
        [0, a/2+w, -b/2-w]
    ])
    No = -N2

    Xo2 = Xo.copy()
    Xo[:,0] -= w/2
    Xo2[:,0] += w/2

    No2 = No.copy()
    No2[:,0] *= -1

    X = np.vstack([X, X2, Xo2, Xo])
    N = np.vstack([N, N2, No2, No])

    Vertices = np.zeros(X.shape[0], dtype=vtype)
    Vertices['a_position'] = X
    Vertices['a_color']    = color
    Vertices['a_normal']   = N

    I = np.array([
        [0,4,1],
        [1,4,5]
        ])

    I = np.concatenate([I+k for k in range(4)])
    I[-2,2] -= 4
    I[-1,0] -= 4
    I[-1,2] -= 4

    I = np.concatenate([I+k*4 for k in range(4)])
    I = np.mod(I, 16)

    return Vertices, np.concatenate(I).astype(np.uint32)

def get_unit_arrow(
        d = 3,
        n = 50,
        ri = 0.06,
        ro = 0.1,
        a = 0.8,
        color = None):
    ''' get a unit length arrow pointing up; scale using u_model'''
    if color is None:
        color = [1, 0, 0, 1]
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    X0 = np.array([[0, 0, 0]])

    if d == 1:
        unit_ring = np.array([np.ones(n), np.cos(th), np.sin(th)])
    elif d == 2:
        unit_ring = np.array([np.sin(th), np.ones(n), np.cos(th)])
    else:
        d = 3
        unit_ring = np.array([np.cos(th), np.sin(th), np.ones(n)])

    X1 = unit_ring.copy()
    X1 = X1 * ri
    X1[d-1] = 0
    X2 = unit_ring.copy()
    X2 = X2 * ri
    X2[d-1] = a
    X3 = unit_ring.copy()
    X3 = X3 * ro
    X3[d-1] = a

    if d == 1:
        XF = np.array([[1, 0, 0]])
    elif d == 2:
        XF = np.array([[0, 1, 0]])
    else:
        XF = np.array([[0, 0, 1]])

    V = np.concatenate([
        X1.T,
        X2.T,
        X3.T,
        X0,
        XF
    ])

    N = np.concatenate([
        np.array([np.cos(th), np.sin(th), np.zeros(n)]).T,
        np.array([np.cos(th), np.sin(th), np.zeros(n)]).T,
        np.array([np.cos(th), np.sin(th), np.zeros(n)]).T,
        np.array([[0, 0, -1]]),
        np.array([[0, 0, 1]])
    ])
    if d == 1:
        N = N[:,[2,0,1]]
    elif d == 2:
        N = N[:,[1,2,0]]

    # first face
    I = np.array([
        [0, 1, n],
        [1,n+1,n]
    ])

    # first segment
    I = np.concatenate([I+k for k in range(n)])
    I[-1,0] -= n
    I[-1,1] -= n
    I[-2,1] -= n

    # arrow perimeter
    I = np.concatenate([I + k for k in range(0, 2*n, n)])

    # bottom
    I_lower = np.array([
        [np.mod(k+1, n), k, V.shape[0]-2] for k in range(n)
    ])

    # top
    I_upper = np.array([
        [k+2*n, k+1+2*n, V.shape[0]-1] for k in range(n)
    ])
    I_upper[-1,1] -= n

    I = np.concatenate([
        I,
        I_lower,
        I_upper
    ])

    Vertices = np.zeros(V.shape[0], dtype=vtype)
    Vertices['a_position'] = V
    Vertices['a_color']    = np.array(color)
    Vertices['a_normal']   = N

    return Vertices, np.concatenate(I).astype(np.uint32)

def get_cube(w = 1.0, x = None, color = None):
    ''' get a cube object'''
    if color is None:
        color = [1,0,0,1]
    Vertices = np.zeros(8, dtype = vtype)
    Vertices['a_position'] = np.array([
        [ 1, 1, 1],
        [-1, 1, 1],
        [-1,-1, 1],
        [ 1,-1, 1],
        [ 1,-1,-1],
        [ 1, 1,-1],
        [-1, 1,-1],
        [-1,-1,-1]
    ]) * w/2
    Vertices['a_normal'] = Vertices['a_position'] * 2/w / np.sqrt(3)
    Vertices['a_color'] = np.array(color)

    if x is not None:
        Vertices['a_position'] += x

    Indices = np.array([
        0,1,2,
        0,2,3,
        0,3,4,
        0,4,5,
        0,5,6,
        0,6,1,
        1,6,7,
        1,7,2,
        7,4,3,
        7,3,2,
        4,7,6,
        4,6,5], dtype=np.uint32)

    return Vertices, Indices

_CACHED_SPHERES = {}
def get_sphere(n=3, r = 1, x = None, color = None) -> Tuple[np.ndarray, np.ndarray]:
    ''' get a sphere with geodesic triangulation '''
    # based on https://stackoverflow.com/questions/17705621/algorithm-for-a-geodesic-sphere
    if color is None:
        color = [1,0,0,1]
    if x is None:
        x = np.array([0., 0., 0.])

    def subdivide(v1, v2, v3, i1, i2, i3, pts, idxs, depth) -> None:
        ''' recursively subdivide triangles on sphere '''
        if depth == 0:
            idxs.append([i1, i3, i2])
            return

        v12 = v1 + v2
        v12 = v12 / np.linalg.norm(v12)
        v23 = v2 + v3
        v23 = v23 / np.linalg.norm(v23)
        v31 = v1 + v3
        v31 = v31 / np.linalg.norm(v31)

        i12 = len(pts)
        i23 = len(pts) + 1
        i31 = len(pts) + 2
        pts.append(v12)
        pts.append(v23)
        pts.append(v31)

        subdivide(v1,  v12, v31, i1,  i12, i31, pts, idxs, depth-1)
        subdivide(v2,  v23, v12, i2,  i23, i12, pts, idxs, depth-1)
        subdivide(v3,  v31, v23, i3,  i31, i23, pts, idxs, depth-1)
        subdivide(v12, v23, v31, i12, i23, i31, pts, idxs, depth-1)

    def initialize(depth=1) -> Tuple[np.ndarray, np.ndarray]:
        ''' initialize and subdivide sphere '''
        X = 0.525731112119133606
        Z = 0.850650808352039932
        verts = [
            [-X, 0.0, Z],
            [ X, 0.0, Z ],
            [-X, 0.0, -Z ],
            [ X, 0.0, -Z ],
            [ 0.0, Z, X ],
            [ 0.0, Z, -X ],
            [ 0.0, -Z, X ],
            [ 0.0, -Z, -X ],
            [ Z, X, 0.0 ],
            [-Z, X, 0.0 ],
            [ Z, -X, 0.0 ],
            [-Z, -X, 0.0 ],
        ]
        faces = [
            [0, 4, 1],
            [ 0, 9, 4 ],
            [ 9, 5, 4 ],
            [ 4, 5, 8 ],
            [ 4, 8, 1 ],
            [ 8, 10, 1 ],
            [ 8, 3, 10 ],
            [ 5, 3, 8 ],
            [ 5, 2, 3 ],
            [ 2, 7, 3 ],
            [ 7, 10, 3 ],
            [ 7, 6, 10 ],
            [ 7, 11, 6 ],
            [ 11, 0, 6 ],
            [ 0, 1, 6 ],
            [ 6, 1, 10 ],
            [ 9, 0, 11 ],
            [ 9, 11, 2 ],
            [ 9, 2, 5 ],
            [ 7, 2, 11 ],
        ]
        verts = [np.array(v) for v in verts]
        pts = verts.copy()
        idxs = []

        for face in faces:
            subdivide(
                *itemgetter(*face)(verts),
                *face,
                pts = pts,
                idxs = idxs,
                depth = depth
                )

        return np.array(pts), np.array(idxs)

    if n in _CACHED_SPHERES:
        X, I = _CACHED_SPHERES[n]
    else:
        X, I = initialize(depth = n)
        _CACHED_SPHERES[n] = (X, I)

    V = np.zeros(X.shape[0], dtype=vtype)
    V['a_position'] = X * r + x
    V['a_normal'] = X
    V['a_color'] = np.array(color)

    return V, np.concatenate(I).astype(np.uint32)

def get_instance_transforms(
        x: np.ndarray,
        r: np.ndarray = None,
        s: np.ndarray = None) -> np.ndarray:
    '''
    takes an array of positions, rotations, and  scales
    position should be Nx3
    rotation should be Nx3x3
    scale    should be N
    scale and rotation are optional
    returns a Nx4x4 array of transformation matrices
    '''
    N = x.shape[0]
    R = np.repeat(np.eye(4)[np.newaxis, :, :], N, axis = 0)

    if r is not None:
        R[:,:3,:3] = r
    if s is not None:
        R[:,:3,:3] *= s[:,np.newaxis, np.newaxis]
    R[:,:3,3] = x
    return R

def _hat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def get_thrust_tf(T: np.ndarray, norm: float = 20) -> np.ndarray:
    ''' helper to get thrust rotation matrix '''
    if T[0] == 0 and T[1] == 0:
        return np.diag([1, 1, T[2]/norm])
    t = T / np.linalg.norm(T)
    b = np.array([0, 0, 1])
    v =-np.cross(t, b)
    s = np.linalg.norm(v)
    c = np.dot(t, b)
    R = np.eye(3) + _hat(v) + _hat(v) @ _hat(v) *(1-c) / s**2
    R = R @ np.diag([1, 1, np.linalg.norm(T) / norm])
    return R.astype(np.float32)

def plot_multiline(
        time: List[np.ndarray],
        data: List[np.ndarray],
        colors: List[bytes],
        labels: List[str],
        title: str,
        t_min: float = None,
        t_max: float = None,
        y_min: float = None,
        y_max: float = None,
        size: Tuple[int, int] = (0,150),
        vbar: float = None,
        show_title: bool = True):
    '''
    helper function to make multi-line plots in imgui
    '''
    if t_min is None:
        t_min = min(tk.min() for tk in time)
    if t_max is None:
        t_max = max(tk.max() for tk in time)
    if y_min is None:
        y_min = min(yk.min() for yk in data if yk is not None)
    if y_max is None:
        y_max = max(yk.max() for yk in data if yk is not None)
    if y_max == y_min:
        y_max += 1e-6
        y_min -= 1e-6

    if size[0] <= 0:
        size = (imgui.core.get_window_content_region_width(), size[1])
    draw_list = imgui.get_window_draw_list()

    org = imgui.core.get_window_position()
    off = imgui.core.get_cursor_pos()

    x0 = org.x + off.x - imgui.core.get_scroll_x()
    y0 = org.y + off.y - imgui.core.get_scroll_y()

    white = imgui.get_color_u32_rgba(1,1,1,1)

    draw_list.add_rect_filled(
        x0,
        y0,
        x0+size[0],
        y0+size[1],
        imgui.get_color_u32_rgba(*[0.13725491, 0.20784314, 0.30980393, 0.7254902 ]),
        5)

    for t, line, color in zip(time, data, colors):
        if line is None:
            continue

        px = (t - t_min) / (t_max - t_min) * size[0] + x0
        py = (line - y_min) / (y_max - y_min)
        py = y0 + 1 + (size[1]-2) * (1 - py)

        draw_list.add_polyline(np.array([px,py]).T.tolist(), color)


    imgui.begin_child(title, size[0], size[1], border=True)
    imgui.end_child()

    if imgui.is_item_hovered():
        imgui.begin_tooltip()

        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()
        tooltip_draw_list = imgui.get_window_draw_list()
        px = org.x + off.x+5
        py = org.y + off.y+2
        dy = 20

        mouse_pos = imgui.core.get_mouse_pos()
        t_tooltip = t_min + (t_max - t_min) * (mouse_pos.x - x0) / size[0]

        tooltip_draw_list.add_text(px + 5, py, white, f't = {t_tooltip:0.2f}')
        py += dy

        no_tooltip_lines = 0
        for t, line, color, label in zip(time, data, colors, labels):
            if t_tooltip < t[0]:
                continue
            if line is None:
                continue

            idx = np.searchsorted(t, t_tooltip)

            if idx < len(line):
                tooltip_draw_list.add_rect_filled(px, py+3 , px+30, py+15, color)
                tooltip_draw_list.add_text(px+35, py, white, f'({line[idx]:9.3f}) {label}')
                py += dy

                no_tooltip_lines += 1

        imgui.begin_child("region", 300, dy*(no_tooltip_lines+1), border=False)
        imgui.end_child()
        imgui.end_tooltip()
    if vbar is not None:
        draw_list.add_line(x0 + size[0]*vbar, y0, x0 + size[0]*vbar, y0 + size[1], white, 2)


    draw_list.add_text(x0, y0, white, f'{y_max:8.2f}')
    draw_list.add_text(x0, y0+size[1]-18, white, f'{y_min:8.2f}')
    draw_list.add_line(x0, y0,x0+60, y0, white)
    draw_list.add_line(x0, y0+size[1],x0+60, y0+size[1], white)

    if show_title:
        draw_list.add_text(x0+102, y0+2, white, str(title))
