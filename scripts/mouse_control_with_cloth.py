import argparse
import math
import os

import numpy as np

import genesis as gs
from genesis.ext.pyrender.camera import OrthographicCamera
from genesis.utils.misc import tensor_to_array
from genesis.utils.raycast import Ray
from genesis.vis.keybindings import MouseButton
from genesis.vis.viewer_plugins import EVENT_HANDLED
from genesis.vis.viewer_plugins.base import ViewerPlugin


class ClothMouseInteractionPlugin(ViewerPlugin):
    """Drag the cloth particle nearest to the mouse ray."""

    def __init__(self, cloth, use_force=False, pick_radius=0.12):
        super().__init__()
        self.cloth = cloth
        self.use_force = use_force
        self.pick_radius = pick_radius
        self.held_particle = None
        self.interact_env_idx = None
        self.env_offset = np.zeros(3, dtype=gs.np_float)
        self.drag_plane = None
        self.last_mouse_pos = (0, 0)

    def build(self, viewer, camera, scene):
        super().build(viewer, camera, scene)
        self.last_mouse_pos = (viewer._viewport_size[0] // 2, viewer._viewport_size[1] // 2)

    def on_mouse_motion(self, x, y, dx, dy):
        self.last_mouse_pos = (x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.last_mouse_pos = (x, y)
        if self.held_particle is not None:
            return EVENT_HANDLED

    def on_mouse_press(self, x, y, button, modifiers):
        if button != MouseButton.LEFT:
            return

        ray_origin, ray_direction = self._screen_position_to_ray(x, y)
        env_idx = self._pick_environment(ray_origin, ray_direction) if self.scene.n_envs > 0 else None
        particle_positions = tensor_to_array(self.cloth.get_particles_pos(envs_idx=env_idx))
        if self.scene.n_envs > 0:
            particle_positions = particle_positions[0]
        if env_idx is None:
            world_positions = particle_positions
        else:
            self.env_offset = self.scene.envs_offset[env_idx].astype(gs.np_float, copy=False)
            world_positions = particle_positions + self.env_offset

        ray_to_particles = world_positions - ray_origin
        ray_distances = ray_to_particles @ ray_direction
        valid = ray_distances > 0.0
        if not np.any(valid):
            return

        closest_points = ray_origin + ray_distances[:, None] * ray_direction
        perpendicular_distances = np.linalg.norm(world_positions - closest_points, axis=1)
        perpendicular_distances[~valid] = np.inf
        particle_idx = int(np.argmin(perpendicular_distances))
        if perpendicular_distances[particle_idx] > self.pick_radius:
            return

        self.held_particle = particle_idx
        self.interact_env_idx = env_idx
        self.drag_plane = (ray_direction, -np.dot(ray_direction, world_positions[particle_idx]))
        return EVENT_HANDLED

    def on_mouse_release(self, x, y, button, modifiers):
        if button == MouseButton.LEFT:
            self.held_particle = None
            self.interact_env_idx = None
            self.env_offset = np.zeros(3, dtype=gs.np_float)
            self.drag_plane = None

    def update_on_sim_step(self):
        if self.held_particle is None or self.drag_plane is None:
            return

        ray_origin, ray_direction = self._screen_position_to_ray(*self.last_mouse_pos)
        plane_normal, plane_offset = self.drag_plane
        denominator = np.dot(plane_normal, ray_direction)
        if abs(denominator) < gs.EPS:
            return

        distance = -(np.dot(plane_normal, ray_origin) + plane_offset) / denominator
        target_world_position = ray_origin + distance * ray_direction
        target_position = target_world_position - self.env_offset

        if self.use_force:
            current_position = tensor_to_array(self.cloth.get_particles_pos(envs_idx=self.interact_env_idx))
            if self.scene.n_envs > 0:
                current_position = current_position[0]
            current_position = current_position[self.held_particle]
            velocity = (target_position - current_position) / self.scene.sim.dt
            self.cloth.set_particles_vel(velocity, [self.held_particle], self.interact_env_idx)
        else:
            self.cloth.set_particles_pos(target_position, [self.held_particle], self.interact_env_idx)

    def _pick_environment(self, ray_origin, ray_direction):
        offsets = np.asarray(self.scene.envs_offset)
        distances = np.linalg.norm(np.cross(offsets - ray_origin, ray_direction), axis=1)
        return int(np.argmin(distances))

    def _screen_position_to_ray(self, x, y):
        viewport_size = self.viewer._viewport_size
        width = float(viewport_size[0])
        height = max(float(viewport_size[1]), 1e-8)
        sx = 2.0 * (float(x) - 0.5 * width) / height
        sy = 2.0 * (float(y) - 0.5 * height) / height

        camera_matrix = self.camera.matrix
        position = camera_matrix[:3, 3]
        forward = -camera_matrix[:3, 2]
        right = camera_matrix[:3, 0]
        up = camera_matrix[:3, 1]

        camera = self.camera.camera
        if isinstance(camera, OrthographicCamera):
            origin = position + right * (sx * float(camera.ymag)) + up * (sy * float(camera.ymag))
            direction = forward / np.linalg.norm(forward)
            return Ray(origin, direction)

        direction = forward + right * (sx * np.tan(0.5 * float(camera.yfov))) + up * (sy * np.tan(0.5 * float(camera.yfov)))
        direction /= np.linalg.norm(direction)
        return Ray(position, direction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mouse interaction with cubes and a deformable cloth.")
    parser.add_argument(
        "--use_force", "-f", action="store_true", help="Move the cloth particle with velocity instead of teleporting it"
    )
    parser.add_argument("--num_envs", "-b", type=int, default=1, help="Number of environments to create")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU backend instead of the GPU backend")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        pbd_options=gs.options.PBDOptions(particle_size=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    for i in range(6):
        angle = i * (2 * math.pi / 6)
        radius = 0.5 + i * 0.1
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(radius * math.cos(angle), radius * math.sin(angle), 0.1 + i * 0.1),
                size=(0.2, 0.2, 0.2),
            )
        )

    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(file="meshes/cloth.obj", pos=(-0.55, 0.0, 0.8), scale=0.7),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(color=(0.9, 0.15, 0.2, 1.0)),
    )

    scene.viewer.add_plugin(ClothMouseInteractionPlugin(cloth, use_force=args.use_force))
    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=args.use_force,
            color=(0.1, 0.6, 0.8, 0.6),
        )
    )

    scene.build(n_envs=args.num_envs)

    is_running = True

    def stop():
        global is_running
        is_running = False

    import genesis.vis.keybindings as kb

    scene.viewer.register_keybinds(kb.Keybind("quit", kb.Key.ESCAPE, kb.KeyAction.RELEASE, callback=stop))

    try:
        while is_running:
            scene.step()
            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")