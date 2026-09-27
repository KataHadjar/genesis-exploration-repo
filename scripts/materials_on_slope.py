import argparse
import numpy as np

import genesis as gs


def build_scene(show_viewer: bool = True):
    gs.init(backend=gs.cpu, precision="32")

    slope_angle_deg = 35.0
    slope_angle = np.deg2rad(slope_angle_deg)
    ramp_pos = np.array((0.0, 0.0, 0.35))
    ramp_size = (3.0, 1.5, 0.15)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            box_box_detection=True,
            enable_joint_limit=True,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.8, -2.8, 1.7),
            camera_lookat=(0.0, 0.0, 0.35),
            camera_fov=46,
        ),
        show_viewer=show_viewer,
    )

    # Shared inclined lane: fixed world support so it stays in place.
    # Tilt the ramp in the X direction so the downhill axis is along X.
    scene.add_entity(
        gs.morphs.Box(
            size=ramp_size,
            pos=ramp_pos,
            euler=(0.0, slope_angle_deg, 0.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            friction=0.5,
            friction_torsional=0.15,
            friction_rolling=0.02,
        ),
    )

    # Side rails: fixed boundaries, not dynamic bodies.
    scene.add_entity(
        gs.morphs.Box(
            size=(3.0, 0.06, 0.25),
            pos=(0.0, 0.75, 0.42),
            euler=(0.0, slope_angle_deg, 0.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            friction=0.5,
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(3.0, 0.06, 0.25),
            pos=(0.0, -0.75, 0.42),
            euler=(0.0, slope_angle_deg, 0.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            friction=0.5,
        ),
    )

    # Different materials: steel, wood, rubber. Each object gets its own lane.
    objects = []
    material_names = ("steel", "wood", "rubber")
    starts = [(-1.0, -0.45), (-1.0, 0.0), (-1.0, 0.45)]
    colors = ((0.2, 0.8, 1.0, 1.0), (0.8, 0.2, 0.2, 1.0), (1.0, 0.65, 0.2, 1.0))
    materials = [
        gs.materials.Rigid(rho=7800.0, friction=0.15, friction_rolling=0.01),
        gs.materials.Rigid(rho=700.0, friction=0.25, friction_rolling=0.02),
        gs.materials.Rigid(rho=1200.0, friction=0.35, friction_rolling=0.03),
    ]

    for (x, y), material_name, mat, color in zip(starts, material_names, materials, colors):
        # Put each cube on the rotated top face, with a small clearance for contact.
        cube_size = 0.14
        surface_z = ramp_pos[2] + np.cos(slope_angle) * ramp_size[2] / 2 - np.tan(slope_angle) * x
        normal_clearance = cube_size / 2 * np.cos(slope_angle) + 0.01
        obj = scene.add_entity(
            gs.morphs.Box(
                size=(cube_size, cube_size, cube_size),
                pos=(x, y, surface_z + normal_clearance),
                euler=(0.0, slope_angle_deg, 0.0),
            ),
            material=mat,
            surface=gs.surfaces.Default(color=color),
        )
        objects.append(obj)

    scene.build()
    return scene, objects, material_names


def main():
    parser = argparse.ArgumentParser(description="Objects with different materials roll down a common slope.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    args = parser.parse_args()

    scene, objects, material_names = build_scene(show_viewer=not args.headless)

    if args.headless:
        for _ in range(args.steps):
            scene.step()
    else:
        step_count = 0
        while scene.viewer.is_alive():
            scene.step()
            scene.viewer.update(force=True)
            step_count += 1
            if args.steps > 0 and step_count >= args.steps:
                break

    # Print positions to confirm they moved down the slope
    print("Final positions and materials:")
    for material_name, obj in zip(material_names, objects):
        pos = obj.get_pos()
        print(f"{material_name}: {np.array(pos).tolist()}")


if __name__ == "__main__":
    main()
