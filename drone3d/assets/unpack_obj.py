''' script used to unpack obj file and save to numpy format '''
import trimesh
import numpy as np
from drone3d.visualization.shaders import vtype

file = 'arena_track_obstacles_multistory.obj'
mesh = trimesh.load(file, force='mesh')

V = np.zeros(np.prod(mesh.faces.shape), dtype = vtype)
V['a_position'] = mesh.vertices[np.concatenate(mesh.faces)]
V['a_color'] = np.array([1,1,0,1])
V['a_normal'] = np.kron(mesh.face_normals, np.array([[1,1,1]]).T)

#np.savez_compressed('obstacles.npz',
#    a_position = V['a_position'],
#    a_normal = V['a_normal'],
#    a_color = np.array([1,1,0,1]))
