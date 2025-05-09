'''
plotting / rendering based on OpenGL and Imgui
'''
from typing import Dict, Tuple, List, Callable
import platform
import time

import numpy as np

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

from drone3d.utils.load_utils import get_assets_file
from drone3d.visualization import glm
from drone3d.pytypes import Position, OrientationQuaternion, BodyAngularVelocity
from drone3d.centerlines.base_centerline import BaseCenterline
from drone3d.visualization.objects import UBOObject, OpenGLObject, Skybox, GroundPlane, gl

def get_font_file(name = 'DejaVuSans.ttf'):
    ''' helper function to get a path to a provided font '''
    filename = get_assets_file(name)
    return filename

class Window():
    '''
    class for creating a window using GLFW,
    adding objects to it to draw with OpenGL,
    and drawing a GUI using imgui
    '''
    window = None
    window_open:bool = True
    window_height:int = 1080
    window_width:int = 1920
    imgui_width:int = 300
    impl = None
    should_close:bool = False

    ubo: UBOObject
    skybox: Skybox

    # camera follow variables
    # unused in rendering here but may be used by child classses
    camera_follow:bool = False

    # mouse drag variables
    _mouse_drag_prev_delta:Tuple[int, int] = (0,0)
    _drag_mice:List[float] = [0,1,2]
    _drag_mouse:int = -1
    _drag_mouse_callbacks: List[Callable] = None

    window_objects: Dict[str, Tuple[bool, OpenGLObject]]
    translucent_window_objects: Dict[str, Tuple[bool, OpenGLObject]]

    # misc interal states
    ground_plane_height: float = 0

    def __init__(self,
            line: BaseCenterline,
            fullscreen = False,
            skybox = True,
            ground_plane=False):
        self.line = line
        self._fullscreen = fullscreen
        self.window_objects = {}
        self.translucent_window_objects = {}
        self._drag_mouse_callbacks = {
            0:self._update_left_mouse_drag,
            1:self._update_right_mouse_drag,
            2:self._update_scroll_mouse_drag
        }

        self._create_imgui()
        self._create_window()
        self._create_imgui_renderer()
        self.ubo = UBOObject()
        self._generate_default_objects(skybox, ground_plane)
        self._reset_camera()
        self.update_projection()

    def draw(self):
        ''' draw the window, both OpenGL and Imgui '''
        if not self.window:
            return True

        # gather events
        glfw.poll_events()
        if not self.window_open:
            # delay as if there were a refresh delay
            time.sleep(1/60)
            return self.should_close

        self.impl.process_inputs()

        # draw imgui
        self._draw_imgui()

        # draw opengl
        self._draw_opengl()

        # renger imgui on top of opengl
        imgui.render()
        self.impl.render(imgui.get_draw_data())

        # push render to window
        glfw.swap_buffers(self.window)

        if glfw.window_should_close(self.window):
            self.should_close = True
        elif self.impl.io.keys_down[glfw.KEY_ESCAPE]:
            self.should_close = True

        return self.should_close

    def add_object(self, name:str, obj: OpenGLObject,
            show:bool = True,
            translucent: bool = False):
        ''' add an opengl style object to the window for drawing '''
        if translucent:
            assert name not in self.translucent_window_objects
            self.translucent_window_objects[name] = [show, obj]
        else:
            assert name not in self.window_objects
            self.window_objects[name] = [show, obj]

    def close(self):
        ''' close the window '''
        if not self.window:
            return
        self.window = None
        glfw.terminate()

    def _create_imgui(self):
        imgui.create_context()

    def _create_window(self):
        if not glfw.init():
            self.window = None
            return

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if platform.system() == 'Darwin':
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        glfw.window_hint(glfw.SAMPLES, 4)

        if self._fullscreen:
            video_mode = glfw.get_video_mode(glfw.get_primary_monitor())
            self.window_width = video_mode.size.width
            self.window_height = video_mode.size.height

        self.window = glfw.create_window(self.window_width,
                                         self.window_height,
                                         "Drone Racing",
                                         glfw.get_primary_monitor() if self._fullscreen else None,
                                         None)

        if not self.window:
            glfw.terminate()
            return

        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_window_size_callback(self.window, self._on_resize)

        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POLYGON_SMOOTH)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.window_width, self.window_height = glfw.get_window_size(self.window)

    def _on_resize(self, window, w, h):
        if self.window and window:
            if w == 0 or h == 0:
                self.window_open = False
                return
            else:
                self.window_open = True
            self.window_width = w
            self.window_height = h
            gl.glViewport(0,0,self.window_width,self.window_height)

            self.update_projection()

    def _create_imgui_renderer(self):
        if self.window:
            self.impl = GlfwRenderer(self.window)
            io = imgui.get_io()

            self.imgui_font = io.fonts.add_font_from_file_ttf(
                get_font_file('DejaVuSans.ttf'), 18)
            self.big_imgui_font = io.fonts.add_font_from_file_ttf(
                get_font_file('DejaVuSans.ttf'), 32)
            self.impl.refresh_font_texture()

    def _generate_default_objects(self, skybox = True, ground_plane = True):
        if self.line is not None:
            line_objects = self.line.generate_texture(self.ubo)
            for name, obj in line_objects.items():
                self.add_object(
                    name,
                    obj,
                    show = True if name == 'Gates' else False,
                    translucent = True if name == 'Local Domain' else False
                )

        if skybox:
            self._add_skybox()
        self._add_ground_plane(show=ground_plane)

    def _add_skybox(self):
        ''' add a skybox to the window '''
        skybox = Skybox()

        self.add_object('Skybox', skybox)
        self.skybox = skybox

    def _add_ground_plane(self, show=True):
        ''' add a ground plane to the window '''
        ground_plane = GroundPlane(self.ubo)
        self.add_object('Ground Plane', ground_plane, show)

    def update_projection(self):
        ''' update the camera projection for the window '''
        ps = glm.perspective(45.0, self.window_width / self.window_height, 0.1, 3000.0).T
        ps = ps.astype(np.float32)
        self._update_projection(ps)

    def _update_projection(self, ps: np.ndarray):
        self.ubo.update_projection(ps)
        self.skybox.update_projection(ps)

    def update_camera_pose(self, u_view=None):
        ''' update the camera pose transform for the window '''
        if u_view is None:
            u_view = np.eye(4)

        if self.camera_follow:

            u_camera = np.eye(4, dtype=np.float32)
            u_camera[:3,-1] = -self.camera_follow_x.to_vec()
            u_camera[:3,:3] = self.camera_follow_q.Rinv()

            u_view = u_camera @ u_view
        else:
            u_view[:3,-1] = -self.free_camera_q.Rinv() @ self.free_camera_x.to_vec()
            u_view[:3,:3] = self.free_camera_q.Rinv()

        u_view = u_view.astype(np.float32)
        self._update_view(u_view)

    def _update_view(self, u_view: np.ndarray):
        self.ubo.update_camera_pose(u_view)
        self.skybox.update_camera_pose(u_view)

    def _draw_opengl(self):
        # pylint: disable=unsupported-binary-operation
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_BLEND)
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

        imgui.set_next_window_position(self.window_width - self.imgui_width, 0)
        imgui.set_next_window_size(300,self.window_height)

        imgui.begin("Vehicle Info", closable = False,
            flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)


        imgui.end()
        self._draw_camera_menu()
        self.draw_extras()

        imgui.pop_font()

    def _draw_camera_menu(self):
        imgui.set_next_window_position(self.window_width - self.imgui_width - 200, 0)
        imgui.set_next_window_size(200, 0) # autosize y
        expanded, _ = imgui.begin("Camera Settings",
            closable = False,
            flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        if expanded:
            if imgui.button("Reset Camera"):
                self._reset_camera()
            if imgui.radio_button('Camera Follow', self.camera_follow):
                self.camera_follow = not self.camera_follow

            imgui.separator()
            imgui.text('Hide / Show:')
            # buttons to hide/show rendered objects
            for obj_name, (draw_obj, _) in self.window_objects.items():
                if imgui.radio_button(obj_name, draw_obj):
                    self.window_objects[obj_name][0] = not draw_obj
            for obj_name, (draw_obj, _) in self.translucent_window_objects.items():
                if imgui.radio_button(obj_name, draw_obj):
                    self.translucent_window_objects[obj_name][0] = not draw_obj

        imgui.end()

    def draw_extras(self):
        ''' function to replace for drawing extra items '''

    def _reset_camera(self):
        self.free_camera_x = Position()
        self.free_camera_q = OrientationQuaternion()
        if self.line is not None:
            self.free_camera_q.from_vec([0.5,0.3,0.3,0.7])
            self.free_camera_q.normalize()
            self.free_camera_x.from_vec(
                (self.line.view_center + self.line.view_scale * self.free_camera_q.e3())
            )
        self.camera_follow_x = Position(xk = 12)
        self.camera_follow_q = OrientationQuaternion()
        self.camera_follow_q.from_vec([-0.3,0.3,0.6,-0.6])
        self.camera_follow_q.normalize()

    def _process_mouse_drag(self):
        active_drag = self._drag_mouse >= 0
        io = imgui.get_io()
        if not io.want_capture_mouse:
            self._update_scroll_mouse_drag(0., -10*io.mouse_wheel)

        if not active_drag:
            for mouse in self._drag_mice:
                if imgui.core.is_mouse_clicked(mouse):
                    if not io.want_capture_mouse:
                        self._drag_mouse = mouse
                        self._mouse_drag_prev_delta = (0,0)
        else:
            if imgui.core.is_mouse_released(self._drag_mouse):
                self._drag_mouse = -1
                return

            drag = imgui.get_mouse_drag_delta(self._drag_mouse)
            dx = drag[0] - self._mouse_drag_prev_delta[0]
            dy = drag[1] - self._mouse_drag_prev_delta[1]
            self._mouse_drag_prev_delta = drag

            self._drag_mouse_callbacks[self._drag_mouse](dx, dy)

        if not self.camera_follow:
            self.update_camera_pose(None)

    def _update_right_mouse_drag(self, dx, dy):
        w = BodyAngularVelocity(w2 = -dx, w1 = -dy, w3 = 0)
        if self.camera_follow:
            qdot = self.camera_follow_q.qdot(w)
            self.camera_follow_q.qr += qdot.qr * 0.002
            self.camera_follow_q.qi += qdot.qi * 0.002
            self.camera_follow_q.qj += qdot.qj * 0.002
            self.camera_follow_q.qk += qdot.qk * 0.002
            self.camera_follow_q.normalize()
        else:
            qdot = self.free_camera_q.qdot(w)
            self.free_camera_q.qr += qdot.qr * 0.002
            self.free_camera_q.qi += qdot.qi * 0.002
            self.free_camera_q.qj += qdot.qj * 0.002
            self.free_camera_q.qk += qdot.qk * 0.002
            self.free_camera_q.normalize()

    def _update_left_mouse_drag(self, dx, dy):
        if self.camera_follow:
            self.camera_follow_x.xi -= dx / 150 * max(1, abs(self.camera_follow_x.xk / 20))
            self.camera_follow_x.xj += dy / 150 * max(1, abs(self.camera_follow_x.xk / 20))
        else:
            scale = self.line.view_scale/1000 if self.line is not None else 5
            self.free_camera_x.from_vec(
                self.free_camera_x.to_vec() +
                (-dx*self.free_camera_q.e1() + dy*self.free_camera_q.e2()) * scale
            )

    def _update_scroll_mouse_drag(self, _, dy):
        if self.camera_follow:
            if abs(self.camera_follow_x.xk) > 3:
                self.camera_follow_x.xk += dy * self.camera_follow_x.xk / 100
            else:
                self.camera_follow_x.xk += dy / 30
        else:
            scale = self.line.view_scale/200 if self.line is not None else 25
            self.free_camera_x.from_vec(
                self.free_camera_x.to_vec() +
                dy*self.free_camera_q.e3() * scale
            )
        self.camera_follow_x.xk = max(0, self.camera_follow_x.xk)
