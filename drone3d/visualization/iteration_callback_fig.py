''' a window for rendering racelines at each iteration '''
# pylint: disable=arguments-differ
from typing import Dict, Callable

import casadi as ca
import imgui
import numpy as np

from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.dynamics.dynamics_model import ParametricDynamicsModel
from drone3d.raceline.base_raceline import BaseRaceline, RacelineResults

from drone3d.visualization.objects import VertexObject, InstancedVertexObject
from drone3d.visualization.opengl_fig import Window
from drone3d.visualization.utils import get_instance_transforms
from drone3d.visualization.raceline_visualization_utils import load_trajectory
from drone3d.visualization.gltf2 import TargetObjectSize, \
    load_quadcopter_vertex_object

class CallbackWindow(ca.Callback, Window):
    ''' window for plotting intermediate solutions '''
    raceline: VertexObject = None
    racers: InstancedVertexObject = None
    running: bool = False
    auto_advance_solver: bool = True
    steps_to_advance: int = 1
    steps_advanced: int = 0
    iteration: int = 0

    def __init__(self, line: BaseCenterline, prob: Dict,
                 unpacker: Callable[[ca.DM], RacelineResults],
                 solver: BaseRaceline):
        ca.Callback.__init__(self)
        self.line = line
        self.unpacker = unpacker
        self.solver = solver

        self.nx = prob['x'].shape[0]
        self.ng = prob['g'].shape[0]
        self.np = 0

        # Initialize internal objects
        self.construct('raceline_callback_fig', {})

        Window.__init__(self, line)

        self._update_raceline(unpacker({'x':solver.solver_w0}), False)

    def preview(self):
        '''
        blocking method that is meant to preview the problem's initial guess 
        and allow the user to move the camera
        '''
        while not self.running:
            self.draw()
            if self.should_close:
                self.close()
                return

    def get_n_in(self):
        ''' required by casadi '''
        return ca.nlpsol_n_out()
    def get_n_out(self):
        ''' required by casadi '''
        return 1
    def get_name_in(self, i):
        ''' required by casadi '''
        return ca.nlpsol_out(i)
    def get_name_out(self, i):
        # pylint: disable=unused-argument
        ''' required by casadi '''
        return "ret"
    def get_sparsity_in(self, i):
        ''' required by casadi '''
        n = ca.nlpsol_out(i)
        if n=='f':
            return ca.Sparsity. scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity(0,0)

    def eval(self, arg):
        ''' called by casadi at each iteration'''
        # check that window is open and should not be closed
        if not self.window:
            return [0]
        if self.should_close:
            self.close()

        # Create dictionary
        darg = {}
        for (i,s) in enumerate(ca.nlpsol_out()):
            darg[s] = arg[i]
        raceline = self.unpacker(darg)

        self.steps_advanced += 1
        self.iteration += 1
        self._update_raceline(raceline)
        return [0]

    def _update_raceline(self, raceline: RacelineResults, draw: bool = True):

        V, I, _, _ = load_trajectory(self.line, raceline, global_frame=raceline.global_frame)
        if self.raceline is None:
            self.raceline = VertexObject(self.ubo, V, I, static_draw=False)
            self.add_object('raceline', self.raceline)
        else:
            self.raceline.setup(V, I)

        if self.racers is None:
            self.racers =  load_quadcopter_vertex_object(
                self.ubo, TargetObjectSize(),
                instanced=True,
                color = raceline.color,
                simple=True)
            self.add_object('racers', self.racers)

        t = np.linspace(0, raceline.time, 100)
        z = raceline.z_interp(t[None]).T
        u = raceline.u_interp(t[None]).T
        if isinstance(self.solver.model, ParametricDynamicsModel):
            x = self.line.fast_p2x(z[:,0], z[:,1], z[:,2]).T
        else:
            x = z[:,:3]

        r = self.solver.model.f_R(z.T, u.T)
        r = r.T.reshape((-1,3,3))
        r = r.transpose((0,2,1))
        R = get_instance_transforms(
            x,
            r = r)
        self.racers.apply_instancing(R)

        if not draw:
            return

        self.draw()
        if not self.auto_advance_solver and self.steps_advanced >= self.steps_to_advance \
                or not self.running:
            self.running = False
            while not self.running:
                self.draw()
                if self.should_close:
                    self.close()
                    return

    def draw_extras(self):
        imgui.set_next_window_position(0,0)
        imgui.set_next_window_size(300, 0)

        imgui.begin("Solver Console",
            closable = False,
            flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR \
                | imgui.WINDOW_NO_COLLAPSE)

        imgui.text(f'Solver Iteration :{self.iteration}')

        run_btn_label = 'Run Solver' if self.auto_advance_solver else 'Advance Solver'
        if imgui.radio_button(run_btn_label, self.running):
            self.running = not self.running
            if self.running:
                self.steps_advanced = 0

        if imgui.radio_button('Auto Advance Solver', self.auto_advance_solver):
            self.auto_advance_solver = not self.auto_advance_solver

        if not self.auto_advance_solver:
            _, self.steps_to_advance = imgui.input_int('Steps', self.steps_to_advance)

        imgui.end()

def main():
    ''' demonstration of callback figure '''
    # pylint: disable=import-outside-toplevel
    from drone3d.centerlines.base_centerline import GateShape
    from drone3d.centerlines.spline_centerline import SplineCenterline, SplineCenterlineConfig
    from drone3d.raceline.point_raceline import GlobalPointRaceline, PointConfig, \
        GlobalRacelineConfig
    x = np.array([-1.1, 9.2, 9.2, -4.5, -4.5, 4.75, -2.8])
    y = np.array([-1.6, 6.6, -4, -6, -6, -0.9, 6.8])
    z = np.array([3.6, 1.0, 1.2, 3.5, 0.8, 1.2, 1.2])

    config = SplineCenterlineConfig(x = np.array([x, y, z]))
    config.closed = True
    config.gate_shape = GateShape.SQUARE
    line = SplineCenterline(config)

    config = GlobalRacelineConfig(verbose = True, N = 50)
    config.closed = line.config.closed
    config.gate_xi = x
    config.gate_xj = y
    config.gate_xk = z
    config.fix_gate_center = False
    config.plot_iterations = True
    drone_config = PointConfig(global_r=True)

    solver = GlobalPointRaceline(
        line,
        config,
        drone_config
    )

    solver.solve()

if __name__ == '__main__':
    main()
