'''
Standard types for drone dynamics, ie. state and input
'''
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import copy
import numpy as np
import scipy.spatial.transform as transform

@dataclass
class PythonMsg:
    '''
    base class for creating types and messages in python
    '''
    def __setattr__(self,key,value):
        '''
        Overloads default atribute-setting functionality
          to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally
          adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self,key):
            raise TypeError (f'Not allowed to add new field "{key}" to class {self}')
        else:
            object.__setattr__(self,key,value)

    def copy(self):
        ''' creates a copy of this class instance'''
        return copy.deepcopy(self)

    def pprint(self, indent = 0):
        ''' a more pretty way to print data '''
        indent = max(indent, 0)

        print(' ' * indent + type(self).__name__)
        indent += 2
        for key in vars(self):
            attr = getattr(self, key)
            if isinstance(attr, PythonMsg):
                attr.pprint(indent = indent)
            else:
                print(' ' * indent + f'{key} : {attr}')

@dataclass
class VectorizablePythonMsg(PythonMsg, ABC):
    ''' structure that can be converted to/from a vector '''
    @abstractmethod
    def to_vec(self) -> np.ndarray:
        ''' convert structure to vector '''

    @abstractmethod
    def from_vec(self, vec: np.ndarray) -> None:
        ''' update structure from vector '''

@dataclass
class Position(VectorizablePythonMsg):
    ''' 3D position in the global frame'''
    xi: float = field(default = 0)
    xj: float = field(default = 0)
    xk: float = field(default = 0)

    def xdot(self, q: 'OrientationQuaternion', v: 'BodyLinearVelocity') -> 'Position':
        ''' position derivative from orientation and body frame velocity'''
        # pylint: disable=line-too-long
        xdot = Position()
        xdot.xi = (1 - 2*q.qj**2 - 2*q.qk**2)*v.v1 + 2*(q.qi*q.qj - q.qk*q.qr)*v.v2 + 2*(q.qi*q.qk + q.qj*q.qr)*v.v3
        xdot.xj = (1 - 2*q.qk**2 - 2*q.qi**2)*v.v2 + 2*(q.qj*q.qk - q.qi*q.qr)*v.v3 + 2*(q.qj*q.qi + q.qk*q.qr)*v.v1
        xdot.xk = (1 - 2*q.qi**2 - 2*q.qj**2)*v.v3 + 2*(q.qk*q.qi - q.qj*q.qr)*v.v1 + 2*(q.qk*q.qj + q.qi*q.qr)*v.v2
        return xdot

    def to_vec(self):
        return np.array([self.xi, self.xj, self.xk])

    def from_vec(self, vec):
        self.xi, self.xj, self.xk = vec

@dataclass
class BodyPosition(VectorizablePythonMsg):
    '''
    3D position in the body frame
    vehicle COM is always at 0,0,0
    '''
    x1: float = field(default = 0)
    x2: float = field(default = 0)
    x3: float = field(default = 0)

    def to_vec(self):
        return np.array([self.x1, self.x2, self.x3])

    def from_vec(self, vec):
        self.x1, self.x2, self.x3 = vec

@dataclass
class BodyLinearVelocity(VectorizablePythonMsg):
    ''' body frame linear velocity '''
    v1: float = field(default = 0)
    v2: float = field(default = 0)
    v3: float = field(default = 0)

    def mag(self):
        ''' magnitutde (speed) '''
        return np.sqrt(self.v1**2 + self.v2**2 + self.v3**2)

    def signed_mag(self):
        ''' magntitude, but negative if moving backwards '''
        return self.mag() * np.sign(self.v1)

    def to_vec(self):
        return np.array([self.v1, self.v2, self.v3])

    def from_vec(self, vec):
        self.v1, self.v2, self.v3 = vec

@dataclass
class BodyAngularVelocity(VectorizablePythonMsg):
    ''' body frame angular velocity '''
    w1: float = field(default = 0)
    w2: float = field(default = 0)
    w3: float = field(default = 0)

    def to_vec(self):
        return np.array([self.w1, self.w2, self.w3])

    def from_vec(self, vec):
        self.w1, self.w2, self.w3 = vec

@dataclass
class BodyLinearAcceleration(VectorizablePythonMsg):
    ''' body frame linear acceleration '''
    a1: float = field(default = 0)
    a2: float = field(default = 0)
    a3: float = field(default = 0)

    def to_vec(self):
        return np.array([self.a1, self.a2, self.a3])

    def from_vec(self, vec):
        self.a1, self.a2, self.a3 = vec

@dataclass
class BodyAngularAcceleration(VectorizablePythonMsg):
    ''' body frame angular acceleration'''
    a1: float = field(default = 0)
    a2: float = field(default = 0)
    a3: float = field(default = 0)

    def to_vec(self):
        return np.array([self.a1, self.a2, self.a3])

    def from_vec(self, vec):
        self.a1, self.a2, self.a3 = vec

@dataclass
class OrientationQuaternion(VectorizablePythonMsg):
    ''' euler symmetric parameters '''
    qi: float = field(default = 0)
    qj: float = field(default = 0)
    qk: float = field(default = 0)
    qr: float = field(default = 1)

    def e1(self):
        '''
        longitudinal basis vector
        points in same direction the vehicle does
        '''
        return np.array([1 - 2*self.qj**2   - 2*self.qk**2,
                          2*(self.qi*self.qj + self.qk*self.qr),
                          2*(self.qi*self.qk - self.qj*self.qr)]).T

    def e2(self):
        '''
        lateral basis vector
        points to left side of vehicle from driver's perspective
        '''
        return np.array([2*(self.qi*self.qj - self.qk*self.qr),
                          1 - 2*self.qi**2   - 2*self.qk**2,
                          2*(self.qj*self.qk + self.qi*self.qr)]).T

    def e3(self):
        '''
        normal basis vector
        points towards top of vehicle
        '''
        return np.array([2*(self.qi*self.qk + self.qj*self.qr),
                          2*(self.qj*self.qk - self.qi*self.qr),
                          1 - 2*self.qi**2    - 2*self.qj**2]).T

    def R(self):
        # pylint: disable=line-too-long
        '''
        rotation matrix
        '''
        return np.array([[1 - 2*self.qj**2 - 2*self.qk**2,       2*(self.qi*self.qj - self.qk*self.qr), 2*(self.qi*self.qk + self.qj*self.qr)],
                         [2*(self.qi*self.qj + self.qk*self.qr), 1 - 2*self.qi**2 - 2*self.qk**2,       2*(self.qj*self.qk - self.qi*self.qr)],
                         [2*(self.qi*self.qk - self.qj*self.qr), 2*(self.qj*self.qk + self.qi*self.qr), 1 - 2*self.qi**2 - 2*self.qj**2      ]])

    def Rinv(self):
        # pylint: disable=line-too-long
        '''
        inverse rotation matrix
        '''
        return np.array([[1 - 2*self.qj**2 - 2*self.qk**2,       2*(self.qi*self.qj + self.qk*self.qr), 2*(self.qi*self.qk - self.qj*self.qr)],
                         [2*(self.qi*self.qj - self.qk*self.qr), 1 - 2*self.qi**2 - 2*self.qk**2,       2*(self.qj*self.qk + self.qi*self.qr)],
                         [2*(self.qi*self.qk + self.qj*self.qr), 2*(self.qj*self.qk - self.qi*self.qr), 1 - 2*self.qi**2 - 2*self.qj**2      ]])

    def norm(self):
        '''
        norm of the quaternion
        '''
        return np.sqrt(self.qr**2 + self.qi**2 + self.qj**2 + self.qk**2)

    def normalize(self):
        '''
        normalize a quaternion

        any orientation quaternion must always be normalized
        this function exists to help ensure that
        '''
        norm = self.norm()
        self.qr /= norm
        self.qi /= norm
        self.qj /= norm
        self.qk /= norm
        return

    def to_vec(self):
        return np.array([self.qi, self.qj, self.qk, self.qr])

    def from_vec(self, vec):
        self.qi, self.qj, self.qk, self.qr = vec

    def from_yaw(self, yaw):
        ''' quaternion from yaw (on a flat euclidean surface)'''
        self.qi = 0
        self.qj = 0
        self.qr = np.cos(yaw/2)
        self.qk = np.sin(yaw/2)

    def to_yaw(self):
        ''' quaternion to yaw (on a flat euclidean surface)'''
        return 2*np.arctan2(self.qk, self.qr)

    def from_mat(self, R):
        ''' update from a rotation matrix '''
        self.from_vec(transform.Rotation.from_matrix(R).as_quat())

    def qdot(self,w: BodyAngularVelocity) -> 'OrientationQuaternion':
        ''' derivative from body frame angular velocity '''
        qdot = OrientationQuaternion()
        qdot.qi =  0.5 * (self.qr * w.w1 + self.qj*w.w3 - self.qk*w.w2)
        qdot.qj =  0.5 * (self.qr * w.w2 + self.qk*w.w1 - self.qi*w.w3)
        qdot.qk =  0.5 * (self.qr * w.w3 + self.qi*w.w2 - self.qj*w.w1)
        qdot.qr = -0.5 * (self.qi * w.w1 + self.qj*w.w2 + self.qk*w.w3)
        return qdot

@dataclass
class ParametricPosition(VectorizablePythonMsg):
    ''' parametric position '''
    s: float = field(default = 0.)
    y: float = field(default = 0.)
    n: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.s, self.y, self.n])

    def from_vec(self, vec):
        self.s, self.y, self.n = vec

@dataclass
class Orientation(VectorizablePythonMsg):
    ''' some orientation measure in 3D '''

    @abstractmethod
    def R(self, v):
        ''' rotation matrix '''

    @abstractmethod
    def from_mat(self, R):
        ''' update from rotation matrix '''

    @abstractmethod
    def to_vec(self):
        ''' vectorize structure variables '''

    @abstractmethod
    def from_vec(self, vec):
        ''' update structure variables '''

@dataclass
class GlobalOrientation(Orientation):
    ''' global orientation '''

@dataclass
class RelativeOrientation(Orientation):
    ''' relative orientation'''

@dataclass
class EulerAngles(VectorizablePythonMsg):
    ''' euler angles '''
    a: float = field(default = 0.)
    b: float = field(default = 0.)
    c: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.a, self.b, self.c])

    def from_vec(self, vec):
        self.a, self.b, self.c = vec

    def R(self):
        ''' get rotation matrix '''
        a = self.a
        b = self.b
        c = self.c
        Ra = np.array([
            [np.cos(a),-np.sin(a),0],
            [np.sin(a), np.cos(a),0],
            [0        , 0        ,1]
        ])
        Rb = np.array([
            [np.cos(b),0, np.sin(b)],
            [0,        1, 0        ],
            [-np.sin(b),0, np.cos(b)]
        ])
        Rc = np.array([
            [1, 0,        0         ],
            [0, np.cos(c),-np.sin(c)],
            [0, np.sin(c), np.cos(c)]
        ])
        return Ra @ Rb @ Rc

    def from_mat(self, R):
        ''' update from a rotation matrix '''
        cba = transform.Rotation.from_matrix(R).as_euler('xyz', degrees=False)
        self.from_vec(np.flip(cba))

@dataclass
class GlobalEulerAngles(EulerAngles, GlobalOrientation):
    ''' global euler angles '''

@dataclass
class RelativeEulerAngles(EulerAngles, RelativeOrientation):
    ''' relative euler angles '''

@dataclass
class GlobalQuaternion(OrientationQuaternion, GlobalOrientation):
    ''' global orientation quaternion'''

@dataclass
class RelativeQuaternion(OrientationQuaternion, RelativeOrientation):
    ''' relative orientation quaternion'''

@dataclass
class RacerConfig(PythonMsg):
    ''' generic racer config '''
    dt: float = field(default = 0.1)
    m: float = field(default = 1.0)
    g: float = field(default = 9.81)

    # velocity drag coefficients) F = -b * v
    b1: float = field(default = 0)
    b2: float = field(default = 0)
    b3: float = field(default = 0)

    global_r: bool = field(default = False)
    collision_radius: float = field(default = 0.3)

@dataclass
class PointConfig(RacerConfig):
    ''' point mass dynamics config '''
    T_max: float = field(default = 32.4)
    T_min: float = field(default = -32.4)
    dT_max: float = field(default = 350)
    dT_min: float = field(default = -350)

@dataclass
class DroneConfig(RacerConfig):
    ''' drone dynamics config '''
    I1: float = field(default = 1.0e-3)
    I2: float = field(default = 1.0e-3)
    I3: float = field(default = 1.7e-3)
    l: float = field(default = 0.15)
    k: float = field(default  = 0.05)

    T_max: float = field(default = 8.1)
    T_min: float = field(default = 0.2)
    dT_max: float = field(default = 20)
    dT_min: float = field(default = -20)

    # angular velocity drag coefficients) K = -w * bw
    bw1: float = field(default = 1e-4)
    bw2: float = field(default = 1e-4)
    bw3: float = field(default = 1e-4)

    w_max: float = field(default = 10)
    w_min: float = field(default = -10)

    use_quat: bool = field(default = False)

@dataclass
class DroneActuation(VectorizablePythonMsg):
    ''' input to drone model'''
    u1: float = field(default = 0.)
    u2: float = field(default = 0.)
    u3: float = field(default = 0.)
    u4: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.u1, self.u2, self.u3, self.u4])

    def from_vec(self, vec):
        self.u1, self.u2, self.u3, self.u4 = vec

@dataclass
class PointActuation(VectorizablePythonMsg):
    ''' input to point mass model'''
    u1: float = field(default = 0.)
    u2: float = field(default = 0.)
    u3: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.u1, self.u2, self.u3])

    def from_vec(self, vec):
        self.u1, self.u2, self.u3 = vec

@dataclass
class RacerState(PythonMsg):
    ''' dynamic state of any racing thing '''
    t: float = field(default = 0.)
    x: Position = field(default = None)
    q: OrientationQuaternion = field(default = None)
    v: BodyLinearVelocity = field(default = None)
    w: BodyAngularVelocity = field(default = None)
    p: ParametricPosition = field(default = None)

    # collision avoidance distance
    d: float = field(default = 0)

    def __post_init__(self):
        if self.x is None:
            self.x = Position()
        if self.q is None:
            self.q = OrientationQuaternion()
        if self.v is None:
            self.v = BodyLinearVelocity()
        if self.w is None:
            self.w = BodyAngularVelocity()
        if self.p is None:
            self.p = ParametricPosition()

@dataclass
class DroneState(RacerState):
    ''' dynamic state of a drone '''
    r: Orientation = field(default = None)
    u: DroneActuation = field(default = None)
    du: DroneActuation = field(default = None)

    def __post_init__(self):
        super().__post_init__()
        if self.r is None:
            self.r = RelativeQuaternion()
        if self.u is None:
            self.u = DroneActuation()
        if self.du is None:
            self.du = DroneActuation()

@dataclass
class PointState(RacerState):
    ''' dynamic state of a point mass '''
    u: PointActuation = field(default = None)
    du: PointActuation = field(default = None)

    def __post_init__(self):
        super().__post_init__()
        if self.u is None:
            self.u = PointActuation()
        if self.du is None:
            self.du = PointActuation()
