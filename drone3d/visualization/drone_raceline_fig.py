'''
code for 3D plotting of one or more racelines
this is an offline visualizer, the racelines are not supposed to change,
just be looped over and over
includes tools to start/stop the animation,
reverse the animation,
manually drag through the animation,
or reverse the animation player

all racelines must have the same configuration for N, K or errors will occur.
'''
# pylint: disable=line-too-long
#TODO - take solver configuration in and add ability to browse?
import time
from typing import Dict, List, Tuple

import numpy as np
import imgui

from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.obstacles.mesh_obstacle import MeshObstacle
from drone3d.raceline.base_raceline import RacelineResults
from drone3d.visualization.utils import get_cmap_rgba, plot_multiline, \
    get_unit_arrow, get_thrust_tf, get_instance_transforms
from drone3d.visualization.raceline_visualization_utils import load_trajectory
from drone3d.visualization.gltf2 import load_quadcopter, TargetObjectSize, \
    load_quadcopter_vertex_object
from drone3d.visualization.objects import VertexObject, gl
from drone3d.visualization.opengl_fig import Window

from drone3d.dynamics.dynamics_model import DynamicsModel, ParametricDynamicsModel
from drone3d.dynamics.drone_models import DroneModel

NUMBER_OF_TRACES = 4

class DroneRacelineWindow(Window):
    '''
    the raceline window class
    '''
    selected_results_index: int = 0
    running: bool = True
    reverse: bool = False
    animation_t:float = 0
    animation_t0:float = -1
    animation_dt:float = 1

    v_min: float = 0
    v_max: float = 1

    camera_follow: bool = True
    camera_follow_modes: List[str] = ['Drone View','Velocity View', 'Centerline View', 'Global View']
    camera_follow_mode: int = 1

    t: np.ndarray
    available_plot_labels: List[str]
    selected_plot_vars: List[int]
    available_plot_vars: Dict['str', np.ndarray]

    raceline_objects: Dict[str, Tuple[bool, Dict[str, Tuple[bool, VertexObject]]]]
    selected_raceline_index: int = 0

    def __init__(self,
            line: BaseCenterline,
            models: List[DynamicsModel],
            results: List[RacelineResults],
            obstacles: Dict[str, MeshObstacle] = None,
            fullscreen = False,
            skybox = True,
            run = True):
        if not isinstance(models, list):
            models = [models]
        if not isinstance(results, list):
            results = [results]

        self.models = models
        self.results = results
        self.obstacles = obstacles

        self.raceline_objects = {}
        self._ensure_unique_result_labels()

        super().__init__(line, fullscreen, skybox)
        self._precompute_parameters()
        self._generate_raceline_objects()

        if run:
            self.run()

    def add_raceline_object(self, result_label: str, name, obj, show = True):
        ''' similar to add_object but creates a reference in self.raceline_objects '''
        if result_label not in self.raceline_objects:
            self.raceline_objects[result_label] = [True, {}]
        self.raceline_objects[result_label][1][name] = [show, obj]

    def run(self):
        ''' run the visualization loop until the window is closed'''
        while not self.should_close:
            self.draw()
        self.close()

    def _ensure_unique_result_labels(self):
        ''' make sure all results have unique labels'''
        used_labels = []
        for result in self.results:
            original_label = result.label
            k = 1
            while result.label in used_labels:
                result.label = original_label + f' ({k})'
                k += 1
            used_labels.append(result.label)

    def _precompute_parameters(self):
        for result in self.results:
            if result.global_frame:
                for state in result.states:
                    self.line.g2lx(state)

        self.t = [np.array([state.t for state in result.states]) for result in self.results]

        s = [np.array([state.p.s for state in result.states]) for result in self.results]
        y = [np.array([state.p.y for state in result.states]) for result in self.results]
        n = [np.array([state.p.n for state in result.states]) for result in self.results]

        v  = [np.array([state.v.mag() for state in result.states]) for result in self.results]
        v1 = [np.array([state.v.v1 for state in result.states]) for result in self.results]
        v2 = [np.array([state.v.v2 for state in result.states]) for result in self.results]
        v3 = [np.array([state.v.v3 for state in result.states]) for result in self.results]

        w1 = [np.array([state.w.w1 for state in result.states]) for result in self.results]
        w2 = [np.array([state.w.w2 for state in result.states]) for result in self.results]
        w3 = [np.array([state.w.w3 for state in result.states]) for result in self.results]

        qi = [np.array([state.q.qi for state in result.states]) for result in self.results]
        qj = [np.array([state.q.qj for state in result.states]) for result in self.results]
        qk = [np.array([state.q.qk for state in result.states]) for result in self.results]
        qr = [np.array([state.q.qr for state in result.states]) for result in self.results]

        d  = [np.array([state.d    for state in result.states]) for result in self.results]

        self.available_plot_vars = {
            'Progress Variable': s,
            'Lateral Offset': y,
            'Normal Offset': n,
            'Speed': v,
            'v1': v1,
            'v2': v2,
            'v3': v3,
            'w1': w1,
            'w2': w2,
            'w3': w3,
            'qi': qi,
            'qj': qj,
            'qk': qk,
            'qr': qr,
        }

        if max(dk.max() for dk in d) > 0:
            self.available_plot_vars['d'] = d


        # add relative orientation traces
        if any(isinstance(model, DroneModel) for model in self.models):
            self.available_plot_vars['ri'] = []
            self.available_plot_vars['rj'] = []
            self.available_plot_vars['rk'] = []
            self.available_plot_vars['rr'] = []

            self.available_plot_vars['ra'] = []
            self.available_plot_vars['rb'] = []
            self.available_plot_vars['rc'] = []

            for result, model in zip(self.results, self.models):
                if isinstance(model, DroneModel):
                    if model.config.use_quat:
                        self.available_plot_vars['ri'].append(np.array([state.r.qi for state in result.states]))
                        self.available_plot_vars['rj'].append(np.array([state.r.qj for state in result.states]))
                        self.available_plot_vars['rk'].append(np.array([state.r.qk for state in result.states]))
                        self.available_plot_vars['rr'].append(np.array([state.r.qr for state in result.states]))
                        self.available_plot_vars['ra'].append(None)
                        self.available_plot_vars['rb'].append(None)
                        self.available_plot_vars['rc'].append(None)
                    else:
                        self.available_plot_vars['ri'].append(None)
                        self.available_plot_vars['rj'].append(None)
                        self.available_plot_vars['rk'].append(None)
                        self.available_plot_vars['rr'].append(None)
                        self.available_plot_vars['ra'].append(np.array([state.r.a for state in result.states]))
                        self.available_plot_vars['rb'].append(np.array([state.r.b for state in result.states]))
                        self.available_plot_vars['rc'].append(np.array([state.r.c for state in result.states]))
                else:
                    self.available_plot_vars['ri'].append(None)
                    self.available_plot_vars['rj'].append(None)
                    self.available_plot_vars['rk'].append(None)
                    self.available_plot_vars['rr'].append(None)
                    self.available_plot_vars['ra'].append(None)
                    self.available_plot_vars['rb'].append(None)
                    self.available_plot_vars['rc'].append(None)

            # remove unused entries
            for label in ['ri','rj','rk','rr','ra','rb','rc']:
                if all(data is None for data in self.available_plot_vars[label]):
                    self.available_plot_vars.pop(label)

        self.available_plot_labels = list(self.available_plot_vars.keys())
        self.selected_plot_vars = list(range(NUMBER_OF_TRACES))

        self.colors = [imgui.get_color_u32_rgba(*result.color) for result in self.results]
        self.labels = [result.label for result in self.results]
        self.show_raceline = [True] * len(self.labels)
        self.times  = [result.time for result in self.results]

        self.t_min = min(tk.min() for tk in self.t)
        self.t_max = max(tk.max() for tk in self.t)
        self.v_min = min(vk.min() for vk in v)
        self.v_max = max(vk.max() for vk in v)

    def _generate_raceline_objects(self):
        for result, model in zip(self.results, self.models):
            racer = load_quadcopter(self.ubo, TargetObjectSize())
            self.add_raceline_object(result.label, 'Racer', racer)

            racers =  load_quadcopter_vertex_object(self.ubo, TargetObjectSize(), instanced=True,
                                                    color = result.color, simple=True)
            t = np.linspace(0, result.time, int(result.time / 0.05))
            z = result.z_interp(t[None]).T
            u = result.u_interp(t[None]).T
            if isinstance(model, ParametricDynamicsModel):
                x = self.line.fast_p2x(z[:,0], z[:,1], z[:,2]).T
            else:
                x = z[:,:3]

            r = model.f_R(z.T, u.T)
            r = r.T.reshape((-1,3,3))
            r = r.transpose((0,2,1))
            R = get_instance_transforms(
                x,
                r = r)
            racers.apply_instancing(R)
            self.add_raceline_object(result.label, 'Racers', racers, show=False)

            V, I, _, _ = load_trajectory(
                self.line, result, global_frame=result.global_frame, closed = result.periodic,
                v_max = self.v_max, v_min = self.v_min)
            raceline = VertexObject(self.ubo, V, I, simple=True)
            self.add_raceline_object(result.label, 'Raceline', raceline)

            V, I = get_unit_arrow(d=3, color = [0, 1, 0, 1])
            vel_arrow = VertexObject(self.ubo, V, I, simple=False)
            self.add_raceline_object(result.label, 'Vel. Arrow', vel_arrow, show=False)

            V, I = get_unit_arrow(d=3)
            if isinstance(model, DroneModel):
                for i in range(4):
                    prop_arrow = VertexObject(self.ubo, V, I, simple=False)
                    self.add_raceline_object(result.label, f'Thrust Arrow {i}', prop_arrow)
            else:
                thrust_arrow = VertexObject(self.ubo, V, I, simple=False)
                self.add_raceline_object(result.label, 'Thrust Arrow', thrust_arrow)

        if self.obstacles is not None:
            for label, obs in self.obstacles.items():
                self.add_object(label, obs.get_vertex_object(self.ubo))

    def _draw_opengl(self):
        # pylint: disable=unsupported-binary-operation
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        for _, (show, objects) in self.raceline_objects.items():
            if show:
                for _, (draw_obj, obj) in objects.items():
                    if draw_obj:
                        obj.draw()
        for _, (draw_obj, obj) in self.window_objects.items():
            if draw_obj:
                obj.draw()

        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        for _, (draw_obj, obj) in self.translucent_window_objects.items():
            if draw_obj:
                obj.draw()
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glDepthMask(gl.GL_TRUE)

    def _draw_imgui(self):
        imgui.new_frame()
        imgui.push_font(self.imgui_font)

        self._process_mouse_drag()

        self._draw_colorbar()


        imgui.set_next_window_position(self.window_width - 300, 0)
        imgui.set_next_window_size(300, self.window_height)
        expanded, _ = imgui.begin("Info", closable = False, flags = imgui.WINDOW_NO_RESIZE)
        self._update_animation_time()
        if expanded:
            self._draw_comparison()
        self._update_vehicles()
        imgui.end()

        self._draw_camera_menu()

        imgui.pop_font()

    def _draw_colorbar(self):
        w = 50
        w_tot = w + 55
        top_pad = 50
        left_pad = 30
        h = self.window_height - top_pad*2

        draw_list = imgui.get_overlay_draw_list()
        C_array = get_cmap_rgba(np.flip(np.arange(h)), 0, h)

        for (C,py) in zip(C_array, range(top_pad, top_pad+h)):
            color = imgui.get_color_u32_rgba(*C)
            draw_list.add_line(left_pad, py, left_pad + w, py, color, thickness=1)

        n = 8
        black = imgui.get_color_u32_rgba(0,0,0,1)
        white = imgui.get_color_u32_rgba(1,1,1,1)

        draw_list.add_rect_filled(left_pad, top_pad-16, left_pad + 160, top_pad, white)
        draw_list.add_text(left_pad +3, top_pad-16, black, 'Vehicle Speed (m/s)')
        for v, py in zip(np.linspace(self.v_max, self.v_min, n), np.linspace(top_pad, top_pad+h, n)):
            draw_list.add_line(left_pad, py, left_pad + w_tot, py, black, thickness=1)
            y_offset = -16 if v == self.v_min else 3
            draw_list.add_rect_filled(left_pad + w +3, py + y_offset, left_pad + w_tot, py + y_offset + 16, white)
            draw_list.add_text(left_pad + w +3, py + y_offset, black, f'{v:6.2f}')

    def _update_animation_time(self):
        if self.running:
            if self.animation_t0 > 0 :
                dt = (time.time() - self.animation_t0) * self.animation_dt
            else:
                dt = 0
            self.animation_t += (-dt if self.reverse else dt)

            if self.line.config.closed:
                if self.animation_t < self.t_min:
                    self.animation_t = self.t_max
                if self.animation_t > self.t_max:
                    self.animation_t = self.t_min
            else:
                if self.animation_t < self.t_min - 2:
                    self.animation_t = self.t_max + 2
                if self.animation_t > self.t_max + 2:
                    self.animation_t = self.t_min - 2
        self.animation_t0 = time.time()

    def _draw_comparison(self):
        imgui.columns(2)
        if imgui.radio_button("Play", self.running):
            self.running = not self.running
        imgui.next_column()
        if imgui.radio_button("Reverse", self.reverse):
            self.reverse = not self.reverse

        imgui.columns(1)

        _, self.animation_t  = imgui.slider_float('Time',self.animation_t, min_value = self.t_min, max_value = self.t_max)
        _, self.animation_dt = imgui.slider_float('Speed',self.animation_dt, min_value = 0.1, max_value = 10,flags=imgui.SLIDER_FLAGS_LOGARITHMIC)

        # draw legend
        dy = 20
        draw_list = imgui.get_window_draw_list()

        white = imgui.get_color_u32_rgba(1,1,1,1)

        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()

        px = org.x + off.x
        py = org.y + off.y

        for color, label, t in zip(self.colors, self.labels, self.times):
            draw_list.add_rect_filled(px, py+3 , px+30, py+15, color)
            if t is not None and label is not None:
                draw_list.add_text(px+35, py, white, f'{label} ({t:0.2f}s)')
            else:
                draw_list.add_text(px+35, py, white, f'{label}')
            py += dy

        imgui.set_cursor_screen_pos((px, py + dy))

        # draw time series info on raceline(s)
        vbar = np.clip((self.animation_t - self.t_min)/(self.t_max - self.t_min), 0, 1)
        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()
        y0 = org.y + off.y
        plot_size = (0, int((self.window_height - y0)/NUMBER_OF_TRACES) - 34)

        generic_args = {
            'time': self.t,
            'colors': self.colors,
            'labels': self.labels,
            't_min': self.t_min,
            't_max': self.t_max,
            'size': plot_size,
            'vbar': vbar,
            'show_title': False
        }

        for k in range(NUMBER_OF_TRACES):
            _, self.selected_plot_vars[k] = \
                imgui.combo(f'##trace {k+1}', self.selected_plot_vars[k], self.available_plot_labels)

            var_label = self.available_plot_labels[self.selected_plot_vars[k]]

            plot_multiline(
                data = self.available_plot_vars[var_label],
                title = f'##trace plot {k}',
                **generic_args)

    def _draw_camera_menu(self):
        imgui.set_next_window_position(self.window_width - self.imgui_width - 200, 0)
        imgui.set_next_window_size(200, 0) # autosize y
        expanded, _ = imgui.begin("Camera Settings",
            closable = False)
        if expanded:
            if imgui.button("Reset Camera"):
                self._reset_camera()
            if imgui.radio_button('Camera Follow', self.camera_follow):
                self.camera_follow = not self.camera_follow
            if self.camera_follow:
                imgui.push_item_width(-1)
                if len(self.results) > 1:
                    _, self.selected_raceline_index = imgui.combo(
                        "##Selected Raceline", self.selected_raceline_index, self.labels)

                _, self.camera_follow_mode = imgui.combo(
                    "##Selected Follow Mode", self.camera_follow_mode, self.camera_follow_modes)
                imgui.pop_item_width()

            imgui.separator()
            imgui.text('Hide / Show:')

            for label, (show, objects) in self.raceline_objects.items():
                if not show:
                    if imgui.radio_button('     ' + label, show):
                        show = not show
                        self.raceline_objects[label][0] = show

                else:
                    imgui.columns(2)
                    imgui.set_column_width(-1, 30)
                    if imgui.radio_button('##' + label, show):
                        show = not show
                        self.raceline_objects[label][0] = show

                    imgui.next_column()
                    expanded, _ = imgui.collapsing_header(label + '##header')
                    if expanded:
                        for name, (draw_obj, _) in objects.items():
                            if imgui.radio_button(name + '##' + label, draw_obj):
                                self.raceline_objects[label][1][name][0] = not draw_obj

                    imgui.columns(1)

            for label, (draw_obj, _) in self.window_objects.items():
                if imgui.radio_button(label, draw_obj):
                    self.window_objects[label][0] = not draw_obj
            for obj_name, (draw_obj, _) in self.translucent_window_objects.items():
                if imgui.radio_button(obj_name, draw_obj):
                    self.translucent_window_objects[obj_name][0] = not draw_obj

            imgui.separator()
            imgui.text('Ground Plane Offset:')
            changed, self.ground_plane_height  = imgui.input_float(
                '##Ground Plane Height',
                self.ground_plane_height)
            if changed:
                mat = np.eye(4,dtype=np.float32)
                mat[2,3] = self.ground_plane_height
                self.window_objects['Ground Plane'][1].update_pose(mat = mat)

        imgui.end()

    def _update_vehicles(self):
        for result, model in zip(self.results, self.models):
            t = np.clip(self.animation_t, 0, result.time)
            z = result.z_interp(t)
            u = result.u_interp(t)

            state = model.get_empty_state()
            model.zu2state(state, z, u)

            racer = self.raceline_objects[result.label][1]['Racer'][1]
            racer.update_pose(state.x, state.q)

            T = model.f_T(z,u)
            thrust = np.linalg.norm(T)

            vg = model.f_vg(z, u)

            # update velocity arrow
            u_model = np.eye(4, dtype=np.float32)
            u_model[:3,-1] = state.x.to_vec()
            u_model[:3,:3] = get_thrust_tf(vg)
            self.raceline_objects[result.label][1]['Vel. Arrow'][1].update_pose(mat = u_model)

            # update thrust arrow(s)
            if isinstance(model, DroneModel):
                for k in range(4):
                    u_model[:3,-1] = state.x.to_vec() + \
                        state.q.e1() * model.config.l * (1 if k in [0,3] else -1) + \
                        state.q.e2() * model.config.l * (1 if k in [0,1] else -1)
                    T_k = T / thrust * u[k]
                    u_model[:3,:3] = get_thrust_tf(T_k, norm = 10)
                    self.raceline_objects[result.label][1][f'Thrust Arrow {k}'][1].update_pose(mat = u_model)

            else:
                u_model[:3,:3] = get_thrust_tf(T)
                self.raceline_objects[result.label][1]['Thrust Arrow'][1].update_pose(mat = u_model)

            if self.camera_follow:
                if result.label == self.labels[self.selected_raceline_index]:
                    u_view = np.eye(4)
                    if self.camera_follow_mode == 0:
                        # drone orientation
                        u_view[:3,-1] = -state.q.Rinv() @ (state.x.to_vec())
                        u_view[:3,:3] = state.q.Rinv()
                    elif self.camera_follow_mode == 1:
                        # velocity orientation
                        th = np.arctan2(vg[1], vg[0])
                        u_view[:2,:2] = np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])
                        u_view[:3,-1] = -u_view[:3,:3] @ (state.x.to_vec())
                    elif self.camera_follow_mode == 2:
                        # centerline view
                        if not isinstance(model, ParametricDynamicsModel):
                            self.line.g2lx(state)
                        es = self.line.p2es(state.p.s)
                        th = np.arctan2(es[1], es[0])
                        u_view[:2,:2] = np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])
                        u_view[:3,-1] = -u_view[:3,:3] @ (state.x.to_vec())
                    elif self.camera_follow_mode == 3:
                        # fixed global orientation
                        u_view[:3,-1] = -state.x.to_vec()
                    else:
                        raise NotImplementedError(f'Unknown camera follow mode {self.camera_follow_mode}')
                    self.update_camera_pose(u_view)
