''' obstacle environment from mesh file '''
from typing import List, Tuple, Dict

import trimesh
from trimesh.proximity import signed_distance, max_tangent_sphere, closest_point
import numpy as np
import casadi as ca
from scipy.spatial import KDTree

from drone3d.utils.load_utils import get_assets_file
from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.visualization.shaders import vtype
from drone3d.visualization.utils import get_sphere, get_instance_transforms, get_circular_gate, \
    get_rectangular_gate, join_vis
from drone3d.visualization.objects import VertexObject, InstancedVertexObject, UBOObject

_DEFAULT_FILENAME = 'arena_track_obstacles_multistory.obj'
class MeshObstacle:
    '''
    mesh-based obstacle
    uses trimesh for loading and distance calculations
        (latter for racelines and collision detection)
    '''
    filename: str
    mesh: trimesh.Trimesh

    def __init__(self,
            filename: str = _DEFAULT_FILENAME,
            color: List[float] = None):
        self.filename = filename
        file = get_assets_file(filename)
        self.mesh = trimesh.load(file, force='mesh')
        self.color = color

        if self.color is None:
            self.color = [1,0,0,1]

    def signed_distance(self, x: np.ndarray) -> np.ndarray:
        '''
        return signed distance, positive for outside of the object
        '''
        return -signed_distance(self.mesh, x)

    def check_line_for_collisions(self, line: BaseCenterline, n:int = 1000) -> float:
        ''' check if a centerline intersects the mesh, return closest distance '''
        s = np.linspace(line.s_min(), line.s_max(), n)[None]
        dist: np.ndarray = self.signed_distance(line.p2xc(s).T)
        return min(dist)

    def compute_plannning_tube(self, line: BaseCenterline, s: np.ndarray, collision_r: float):
        '''
        comput obstacle-free tube given a centerline and path lengths along it
        to compute at
        '''

        x = np.array([line.p2xc(sk) for sk in s])
        ey = np.array([line.p2ey(sk) for sk in s])
        en = np.array([line.p2en(sk) for sk in s])

        ball_center, ball_r, ball_tangent_pts = self.search_largest_sphere(
            x,
            ey,
            en)

        ball_y = np.sum((ball_center - x) * ey, axis = 1)
        ball_n = np.sum((ball_center - x) * en, axis = 1)

        ball_p = np.array([s, ball_y, ball_n]).T

        return ObstacleFreeTube(
            line,
            ball_center,
            ball_r,
            ball_tangent_pts,
            ball_p,
            collision_r)

    def check_for_collisions(self, x: np.ndarray, sep_radius: float = 0.3) -> bool:
        '''
        check points
        x must have shape (nx3)
        '''
        dist: np.ndarray = self.signed_distance(x)
        return (dist >= sep_radius).all()

    def get_empty_sphere(self, x0: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        find the largest empty sphere near the given point
        returns the center position and radius of the sphere
          and the points used to begin sphere search from
        x0 must have shape (nx3)
        '''
        pts, dist, _ = closest_point(self.mesh, x0)
        n = x0 - pts
        n = n / np.linalg.norm(n, axis = 1)[:,np.newaxis]
        x, r = max_tangent_sphere(self.mesh, pts, inwards = False, normals = n)

        # remove points that failed to converge with earlier points
        idxs = np.argwhere(1 - np.isfinite(r))
        r[idxs] = dist[idxs]
        x[idxs] = x0[idxs]

        # remove points that got worse
        idxs = np.argwhere(dist > r)
        r[idxs] = dist[idxs]
        x[idxs] = x[idxs]
        return x, r, pts

    def search_largest_sphere(self, x0: np.ndarray, ey: np.ndarray, en: np.ndarray,
            r_max: float = 0.5, nr: int = 5, nth: int = 8) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        brute force search for largest empty sphere near a point
        x0, ey, and en must have shape (nx3)
        '''
        d0 = self.signed_distance(x0)
        r = np.linspace(r_max, 0, nr, endpoint=False)
        th = np.linspace(0, 2*np.pi, nth, endpoint = False)

        R, TH = np.meshgrid(r,th)
        R = R.reshape(-1)
        TH = TH.reshape(-1)

        X = np.kron(x0, np.ones((R.shape[0], 1))) + \
            np.kron(ey, (R*np.cos(TH))[:,np.newaxis]) + \
            np.kron(en, (R*np.sin(TH))[:,np.newaxis])

        D = self.signed_distance(X).reshape((-1, nr*nth)).T
        idxs = D.argmax(axis = 0)
        dn = D.max(axis = 0)
        rn = R[idxs]
        thn = TH[idxs]

        xn = x0 + rn[:,np.newaxis] * \
            (ey * np.cos(thn[:,np.newaxis]) + en * np.sin(thn[:,np.newaxis]))

        x = xn.copy()
        r = dn.copy()
        x[d0 >= dn] = x0[d0 >= dn]
        r[d0 >= dn] = d0[d0 >= dn]

        pts, _, _ = closest_point(self.mesh, x)

        return x, r, pts

    def get_vertex_object(self, ubo: UBOObject) -> VertexObject:
        ''' get a 3D renderable representation of this mesh '''

        V = np.zeros(np.prod(self.mesh.faces.shape), dtype = vtype)
        V['a_position'] = self.mesh.vertices[np.concatenate(self.mesh.faces)]
        V['a_color'] = np.array(self.color)
        if self.mesh.vertex_normals is not None:
            V['a_normal'] = self.mesh.vertex_normals[np.concatenate(self.mesh.faces)]
        else:
            V['a_normal'] = np.kron(self.mesh.face_normals, np.array([[1,1,1]]).T)
        if self.filename == _DEFAULT_FILENAME:
            I = np.arange(V.shape[0])
            V, I = join_vis(((V, I), _add_default_extra_gates()))
            return VertexObject(ubo, V, I)
        return VertexObject(ubo, V)


def _add_default_extra_gates():
    extra_gate_x = np.array([
        [ 2.95,  1.25, 2.815],
        [-2.75, -0.08, 2.815],
        [-4.70, -6.43, 2.815],
        [ 1.57, -6.43, 2.815]
    ])
    extra_gate_w = np.array([
        [1.68, 2.0, 0.08],
        [1.68, 2.0, 0.08],
        [1.68, 2.0, 0.08],
        [3.43, 2.0, 0.08]
    ])
    extra_gate_R = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ])

    Vertices = None
    Indices = None
    for x,w in zip(extra_gate_x, extra_gate_w):
        V, I = get_rectangular_gate(*w)

        V['a_position'] = V['a_position'] @ extra_gate_R + x
        V['a_normal'] = V['a_normal'] @ extra_gate_R

        if Vertices is None or Indices is None:
            Vertices = V
            Indices = I
        else:
            Indices = np.concatenate([Indices, I + len(Vertices)])
            Vertices = np.concatenate([Vertices, V])
    return Vertices, Indices


class ObstacleFreeTube:
    '''
    class to store, use, and render info on an obstacle-free tube
    '''
    def __init__(self,
            line: BaseCenterline,
            ball_center: np.ndarray,
            ball_r: np.ndarray,
            ball_tangent_pts: np.ndarray,
            ball_p: np.ndarray,
            collision_r: float):
        self.line = line
        self.ball_center = ball_center
        self.ball_r = ball_r
        self.ball_tangent_pts = ball_tangent_pts
        self.ball_p = ball_p
        self.ball_kd_tree = KDTree(ball_p[:,:1])
        self.collision_r = collision_r

    def add_constraints_parametric(self, s: float, z: ca.SX, g, ubg, lbg):
        '''
        add constraints for a given state at path length s
        for a parametric raceline
        '''

        _, ball_idx = self.ball_kd_tree.query(s)

        delta_y = self.ball_p[ball_idx, 1]
        delta_n = self.ball_p[ball_idx, 2]
        available_r  = np.maximum(
            self.ball_r[ball_idx] - self.collision_r,
            0.01)

        r_sq = (z[1] - delta_y)**2 + (z[2] - delta_n)**2

        g += [r_sq]
        ubg += [available_r**2]
        lbg += [-np.inf]

    def get_vertex_objects(self, ubo: UBOObject) -> Dict[str, VertexObject]:
        ''' generate drawables for setup info '''
        obs_free_balls = InstancedVertexObject(ubo, *get_sphere(n=3, color=[0,0,1,1]))
        R = get_instance_transforms(
            self.ball_center,
            s = self.ball_r)
        obs_free_balls.apply_instancing(R)

        planning_tube = InstancedVertexObject(ubo,
            *get_circular_gate(n = 100, ro=1,ri=0.9,w=0.1,color=[0,1,0,1]))
        r = self.line.p2Rp(self.ball_p[:,0:1].T)
        r = r.T.reshape((-1,3,3))
        r = r.transpose((0,2,1))
        R = get_instance_transforms(
            self.ball_center,
            r = r,
            s = np.maximum(self.ball_r - self.collision_r, 0.01))
        planning_tube.apply_instancing(R)

        ball_centers = InstancedVertexObject(ubo, *get_sphere(n = 2, color = [0,0,1,1]))
        R = get_instance_transforms(
            self.ball_center,
            s = np.ones(self.ball_center.shape[0]) * 0.05)
        ball_centers.apply_instancing(R)

        tangent_pts = InstancedVertexObject(ubo, *get_sphere(n = 2, color = [1,0,0,1]))
        R = get_instance_transforms(
            self.ball_tangent_pts,
            s = np.ones(self.ball_center.shape[0]) * 0.05)
        tangent_pts.apply_instancing(R)

        return {
            'Planning Tube': planning_tube,
            'Free-Space Spheres': obs_free_balls,
            'Sphere Centers': ball_centers,
            'Sphere Contact Points': tangent_pts,
        }


if __name__ == '__main__':
    import os
    import time
    import pickle
    import coacd
    from drone3d.visualization.opengl_fig import Window
    parts_path = get_assets_file('arena_track_convex_decomp.pkl')

    if not os.path.exists(parts_path):

        obs = MeshObstacle()
        mesh = coacd.Mesh(obs.mesh.vertices, obs.mesh.faces)
        t0 = time.time()
        parts = coacd.run_coacd(
            mesh,
            threshold=0.01,
            preprocess_resolution=50,
            mcts_iterations=200,
            mcts_nodes=40,
            mcts_max_depth=5)
        print(f'Computed Convex Decomp in {time.time() - t0:0.2f} seconds')
        with open(parts_path, 'wb+') as f:
            pickle.dump(parts, f)
    else:
        with open(parts_path, 'rb') as f:
            parts = pickle.load(f)

    window = Window(None)

    for k, (verts, inds) in enumerate(parts):
        mesh = trimesh.Trimesh(verts, inds)
        V = np.zeros(np.prod(mesh.faces.shape), dtype = vtype)
        V['a_position'] = mesh.vertices[np.concatenate(mesh.faces)]
        V['a_color'] = [*np.random.uniform(low = 0, high = 1, size=3), 1.0]
        V['a_normal'] = np.kron(mesh.face_normals, np.array([[1,1,1]]).T)


        window.add_object(f'block {k}',
            VertexObject(window.ubo, V, np.concatenate(inds).astype(np.uint32)))

    while not window.draw():
        pass
