''' centerline with cublic spline center and cubic spline lateral offset '''
from dataclasses import dataclass, field
from typing import Callable, Tuple, Union
from enum import Enum

import casadi as ca
import numpy as np
import scipy.interpolate

from drone3d.utils.interp import spline_interpolant
from drone3d.centerlines.base_centerline import BaseCenterline, BaseCenterlineConfig, BaseSymRep

class SplineRyFitOptions(Enum):
    ''' options for fitting ry when unspecified '''
    TORSION_FREE=0
    PLANAR = 1
    PRINCIPAL_CURVATURE = 2

@dataclass
class SplineCenterlineConfig(BaseCenterlineConfig):
    '''
    config for the racetrack
    '''
    s: Union[np.ndarray, None] = field(default = None)
    x: np.ndarray = field(default = None)
    ry: Union[np.ndarray, None] = field(default = None)
    ry_fit_method: SplineRyFitOptions = field(default = SplineRyFitOptions.PLANAR)

    def __post_init__(self):
        if self.x is None:
            self.x = np.array([
                [0,0,0],
                [10,0,0],
                [10,10,0],
                [0,10,0],
                [0,0,0]
            ]).T


class SplineCenterline(BaseCenterline):
    ''' centerline parameterized by a cubic spline '''
    config: SplineCenterlineConfig

    # centerline symbolic interpolation- all 1 input 3 output
    xc:   Callable[[ca.SX], ca.SX]
    xcs:  Callable[[ca.SX], ca.SX]
    xcss: Callable[[ca.SX], ca.SX]

    # lateral direction symbolic interpolation- all 1 input 3 output
    ry:  Callable[[ca.SX], ca.SX]
    rys:  Callable[[ca.SX], ca.SX]

    # scipy splines for the centerline and lateral direction
    # meant for fast, non-symbolic evaluation
    _center_spline: Callable[..., np.ndarray] = None
    _lateral_spline: Callable[..., np.ndarray] = None

    def __init__(self, config: SplineCenterlineConfig):
        if not isinstance(config.gate_s, np.ndarray) and isinstance(config.s, np.ndarray):
            config.gate_s = config.s
        super().__init__(config)

    def fast_p2x(self, s,y,n):
        '''
        fast local to global conversion of position using Scipy splines
        this is faster for generating surface textures, since it is vectorized,
        whereas in a CasADi function it is not.
        In optimization problems, self.pose2xp should be used.
        '''
        xc = self._center_spline(s).T.squeeze()

        xcs = self._center_spline(s, 1).T
        es = xcs / np.linalg.norm(xcs, axis = 0)
        es = es.squeeze()

        ry = self._lateral_spline(s).T.squeeze()
        ry = ry  - es * (es * ry).sum(axis = 0)
        ey = ry / np.linalg.norm(ry, axis = 0)
        en = np.cross(es.T, ey.T).T.squeeze()
        xp = xc + ey * y + en * n
        return xp

    def fast_p2ey(self, s):

        xcs = self._center_spline(s, 1).T
        es = xcs / np.linalg.norm(xcs, axis = 0)
        es = es.squeeze()

        ry = self._lateral_spline(s).T.squeeze()
        ry = ry  - es * (es * ry).sum(axis = 0)
        ey = ry / np.linalg.norm(ry, axis = 0)
        return ey

    def fast_p2en(self, s):

        xcs = self._center_spline(s, 1).T
        es = xcs / np.linalg.norm(xcs, axis = 0)
        es = es.squeeze()

        ry = self._lateral_spline(s).T.squeeze()
        ry = ry  - es * (es * ry).sum(axis = 0)
        ey = ry / np.linalg.norm(ry, axis = 0)
        en = np.cross(es.T, ey.T).T.squeeze()
        return en

    def _fill_in_s(self):
        if self.config.s is None:
            self.config.s = np.arange(self.config.x.shape[1]) * 1
            if self.config.gate_s is None:
                self.config.gate_s = self.config.s
        self.config.s_max = self.config.s.max()
        self.config.s_min = self.config.s.min()

    def _fill_in_ry(self):
        if self.config.ry is not None:
            s_grid = self.config.s
            ry_grid = self.config.ry
        elif self.config.ry_fit_method == SplineRyFitOptions.TORSION_FREE:
            s_grid, ry_grid = self._fill_in_ry_torsion_free()
        elif self.config.ry_fit_method == SplineRyFitOptions.PLANAR:
            s_grid, ry_grid = self._fill_in_ry_planar()
        elif self.config.ry_fit_method == SplineRyFitOptions.PRINCIPAL_CURVATURE:
            s_grid, ry_grid = self._fill_in_ry_principal_curvature()
        else:
            raise NotImplementedError(f'Unhandled ry fit option: {self.config.ry_fit_method}')

        if np.linalg.norm(ry_grid[0] - ry_grid[-1]) < 1e-3:
            ry_grid[-1] = ry_grid[0]
            self.cleanly_closed = True
        else:
            self.cleanly_closed = False

        bc_type = 'periodic' if self.cleanly_closed else 'not-a-knot'
        self._lateral_spline = scipy.interpolate.CubicSpline(s_grid, ry_grid, bc_type=bc_type)

        ryi = spline_interpolant(s_grid, ry_grid[:,0], extrapolate = 'linear',
            bc_type = bc_type,
            fast = False)
        ryj = spline_interpolant(s_grid, ry_grid[:,1], extrapolate = 'linear',
            bc_type = bc_type,
            fast = False)
        ryk = spline_interpolant(s_grid, ry_grid[:,2], extrapolate = 'linear',
            bc_type = bc_type,
            fast = False)

        s = ca.SX.sym('s')
        ry = ca.vertcat(ryi(s), ryj(s), ryk(s))
        self.ry   = ca.Function('ry',   [s], [ry])
        self.rys  = ca.Function('rys',  [s], [ca.jacobian(self.ry(s), s)])

    def _fill_in_ry_planar(self) -> Tuple[np.ndarray, np.ndarray]:
        s_fit = np.linspace(self.s_min(), self.s_max(), 100)

        # interpolate yaw of centerline finely for continuity
        es = self._center_spline(s_fit, 1).T
        th = np.arctan2(es[1], es[0])
        for k in range(1, len(th)):
            while th[k] - th[k-1] > np.pi:
                th[k] -= 2*np.pi
            while th[k-1] - th[k] > np.pi:
                th[k] += 2*np.pi
        th = th + np.pi/2
        thc = scipy.interpolate.CubicSpline(s_fit, th)

        # interpolate yaw coarsely for smooth transition between waypoints
        s_waypoints = self.config.s
        th_waypoints = thc(s_waypoints)
        thc = scipy.interpolate.CubicSpline(s_waypoints, th_waypoints)
        # then resample
        th_fit = thc(s_fit)

        ry_fit = np.array([np.cos(th_fit), np.sin(th_fit), th_fit*0])
        if self.config.closed:
            ry_fit[:,-1] = ry_fit[:,0]

        return s_fit, ry_fit.T

    def _fill_in_ry_torsion_free(self) -> Tuple[np.ndarray, np.ndarray]:
        s_grid = np.linspace(self.s_min(), self.s_max(), 100)
        s = ca.SX.sym('s')

        xc  = self.xc(s)
        xc  = ca.Function('x',   [s], [xc])
        dxc = ca.Function('dx',  [s], [ca.jacobian(xc(s), s)])

        es = dxc(s) / ca.norm_2(dxc(s))
        ey = ca.SX.sym('ey', 3)
        des = ca.jacobian(es, s)
        kg = des.T @ ey

        dey = -kg * es

        ode    = {
            'x':ca.vertcat(s,ey),
            'ode':ca.vertcat(1,dey)}
        config = {
            'max_step_size':(s_grid[1] - s_grid[0])}

        try:
            # try integrator setup for casadi >= 3.6.0
            xint = ca.integrator('zint','idas',ode, self.s_min(), s_grid, config)
        except NotImplementedError:
            config['t0'] = self.s_min()
            config['tf'] = self.s_max()
            config['grid'] = s_grid
            config['output_t0'] = True
            xint = ca.integrator('zint','idas',ode, config)

        dx0 = np.array(dxc(self.s_min())).squeeze()
        ey0 = np.array([-dx0[1], dx0[0], 0])
        ey0 = ey0 / np.linalg.norm(ey0)

        sol = np.array(xint(x0 = [self.s_min(),*ey0])['xf'])
        s_grid = sol[0]
        ry_grid = sol[1:].T

        return s_grid, ry_grid

    def _fill_in_ry_principal_curvature(self) -> Tuple[np.ndarray, np.ndarray]:
        s_fit = self.config.s
        es = self._center_spline(s_fit, 1)
        en = self._center_spline(s_fit, 2)

        es = es / np.linalg.norm(es, axis = 1)[:,np.newaxis]
        en = en - es * (es * en).sum(axis = 1)[:,np.newaxis]
        en = en / np.linalg.norm(en, axis = 1)[:,np.newaxis]

        ey = -np.cross(en, es)

        return s_fit, ey

    def _setup_interp(self):
        if self.config.closed:
            if not (self.config.x[:,0] == self.config.x[:,-1]).all():
                self.config.x = np.hstack([self.config.x, self.config.x[:,0:1]])

        self._fill_in_s()

        bc_type = 'not-a-knot' if not self.config.closed else 'periodic'
        self._center_spline = \
            scipy.interpolate.CubicSpline(self.config.s, self.config.x.T, bc_type = bc_type)

        xi = spline_interpolant(self.config.s, self.config.x[0], extrapolate = 'linear',
            bc_type = bc_type,
            fast = False)
        xj = spline_interpolant(self.config.s, self.config.x[1], extrapolate = 'linear',
            bc_type = bc_type,
            fast = False)
        xk = spline_interpolant(self.config.s, self.config.x[2], extrapolate = 'linear',
            bc_type = bc_type,
            fast = False)

        s = ca.SX.sym('s')

        xc = ca.vertcat(xi(s), xj(s), xk(s))
        self.xc   = ca.Function('xc',   [s], [xc])
        self.xcs  = ca.Function('xcs',  [s], [ca.jacobian(self.xc(s), s)])
        self.xcss = ca.Function('xcss', [s], [ca.jacobian(self.xcs(s), s)])

        s_grid = np.linspace(self.s_min(), self.s_max(), self.config.N_grid)
        x_grid = self._center_spline(s_grid)
        self.xc_grid = np.concatenate([s_grid[:,np.newaxis], x_grid], axis = 1)

        self._fill_in_ry()

    def _compute_sym_rep(self):
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')
        p = ca.vertcat(s,y,n)

        xc   = ca.SX.sym('xc',   3)
        xcs  = ca.SX.sym('xcs',  3)
        xcss = ca.SX.sym('xcss', 3)

        ry = ca.SX.sym('ry', 3)
        rys = ca.SX.sym('rys', 3)

        es = xcs / ca.norm_2(xcs)
        ey = ry - es * (es.T @ ry)
        ey = ey / ca.norm_2(ey)
        en = ca.cross(es, ey)

        x = xc + y*ey + n*en

        # compute curvature of the centerline
        one = ca.vertcat(
            ca.horzcat(xcs.T @ es, xcs.T @ ey),
            ca.horzcat(ry.T @ es , ry.T @ ey )
        )
        kyks = ca.inv(one) @ ca.vertcat(xcss.T @ en, rys.T @ en) / ca.norm_2(xcs)
        ks = kyks[1]
        ky = -kyks[0]
        kn = - ca.cross(xcss, xcs).T @ en / ca.norm_2(xcs)**3


        param_terms = ca.vertcat(xc, xcs, xcss, ry, rys)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.xc(s),
            self.xcs(s),
            self.xcss(s),
            self.ry(s),
            self.rys(s))

        f_param_terms = ca.Function('param_terms',[s], [param_terms_explicit])

        self.sym_rep = BaseSymRep(
            x = x,
            xc = xc,
            p = p,
            es = es,
            ey = ey,
            en = en,
            ks = ks,
            ky = ky,
            kn = kn,
            mag_xcs = ca.norm_2(xcs),
            param_terms=param_terms,
            f_param_terms=f_param_terms
        )


def _main():
    x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8])
    y = np.array([-1.6, 6.6, -4, -6, -6, -0.9, 6.8])
    z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2])

    x = np.array([0, 5, 0, -5,   0,  5,  0, -5])
    y = np.array([0, 1, 2,   1,  0,  -1,  -2, -1])
    z = np.array([10, 5, 0, -5, -10, -5, 0, 5])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.ry_fit_method = SplineRyFitOptions.PRINCIPAL_CURVATURE
    cent = SplineCenterline(config)
    cent.preview_centerline()

if __name__ == '__main__':
    _main()
