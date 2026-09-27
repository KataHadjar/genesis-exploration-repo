import argparse
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import genesis as gs


# ============================================================
# Parameters
# ============================================================

CUP_POS = (-0.04, 0.0, 0.05)

# Fill most of the wider and taller cup, leaving about 1 mm of headspace.
WATER_SIZE = (0.054, 0.054, 0.100)
WATER_CENTER = (
    CUP_POS[0],
    CUP_POS[1],
    CUP_POS[2] + 0.004 + WATER_SIZE[2] / 2 + 0.001,
)

# Robot starts to the left of the cup
ROBOT_POS = (-0.35, 0.0, 0.0)
PLACE_OFFSET = np.array([0.15, 0.0, 0.0], dtype=np.float32)
ARM_DOFS = np.arange(7)
FINGER_DOFS = np.array([7, 8])
GRIP_OPEN = 0.04
GRIP_HOLD_FORCE = -6.0
SIM_DT = 0.004
HAND_ABOVE_CUP_CENTER = 0.090


# ============================================================
# Cup URDF
# ============================================================
#
# The cup consists of:
#   - bottom
#   - front wall
#   - back wall
#   - left wall
#   - right wall
#
# All parts are connected by fixed joints.
# Genesis merges the fixed links into one rigid body.
#
# No water_cup.obj is used as collision geometry.
# ============================================================

CUP_URDF = """<?xml version="1.0"?>
<robot name="cup">

  <link name="bottom">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.08"/>
      <inertia
        ixx="0.0001"
        ixy="0"
        ixz="0"
        iyy="0.0001"
        iyz="0"
        izz="0.0001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
      <box size="0.07 0.07 0.008"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.06 0.06 0.008"/>
      </geometry>
    </collision>
  </link>


  <link name="front">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.025"/>
      <inertia
        ixx="0.00001"
        ixy="0"
        ixz="0"
        iyy="0.00001"
        iyz="0"
        izz="0.00001"/>
    </inertial>

    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
      <!-- front/back walls: long along X, thin along Y -->
      <box size="0.07 0.007 0.11"/>
      </geometry>
    </collision>
  </link>


  <link name="back">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.025"/>
      <inertia
        ixx="0.00001"
        ixy="0"
        ixz="0"
        iyy="0.00001"
        iyz="0"
        izz="0.00001"/>
    </inertial>

    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
      <box size="0.07 0.007 0.11"/>
      </geometry>
    </collision>
  </link>


  <link name="left">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.025"/>
      <inertia
        ixx="0.00001"
        ixy="0"
        ixz="0"
        iyy="0.00001"
        iyz="0"
        izz="0.00001"/>
    </inertial>

    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
      <!-- left/right walls: thin along X, long along Y -->
      <box size="0.007 0.07 0.11"/>
      </geometry>
    </collision>
  </link>


  <link name="right">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.025"/>
      <inertia
        ixx="0.00001"
        ixy="0"
        ixz="0"
        iyy="0.00001"
        iyz="0"
        izz="0.00001"/>
    </inertial>

    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
      <box size="0.007 0.07 0.11"/>
      </geometry>
    </collision>
  </link>


  <!-- Front wall -->
  <joint name="bottom_to_front" type="fixed">
    <parent link="bottom"/>
    <child link="front"/>
    <origin xyz="0 -0.0315 0.051"/>
  </joint>


  <!-- Back wall -->
  <joint name="bottom_to_back" type="fixed">
    <parent link="bottom"/>
    <child link="back"/>
    <origin xyz="0 0.0315 0.051"/>
  </joint>


  <!-- Left wall -->
  <joint name="bottom_to_left" type="fixed">
    <parent link="bottom"/>
    <child link="left"/>
    <origin xyz="-0.0315 0 0.051"/>
  </joint>


  <!-- Right wall -->
  <joint name="bottom_to_right" type="fixed">
    <parent link="bottom"/>
    <child link="right"/>
    <origin xyz="0.0315 0 0.051"/>
  </joint>

</robot>
"""


def make_cup_urdf(cup_mesh_path):
    """Render the hollow cup mesh while retaining simple collision boxes."""
    urdf = re.sub(r"\s*<visual>.*?</visual>", "", CUP_URDF, flags=re.DOTALL)
    visual = f"""
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry><mesh filename="{cup_mesh_path}"/></geometry>
      <material name="porcelain_white"><color rgba="0.97 0.97 0.94 1.0"/></material>
    </visual>
"""
    return urdf.replace('<link name="bottom">', '<link name="bottom">' + visual, 1)


# ============================================================
# Robot motion helper
# ============================================================

def move_ee(scene, robot, target_pos, target_quat, duration=2.0):
    """
    Move the Franka end-effector using IK + joint position control.

    Only the first 7 DOFs are controlled.
    The two gripper DOFs remain untouched.
    """

    q, ik_error = robot.inverse_kinematics(
        link=robot.get_link("hand"),
        pos=np.array(target_pos),
        quat=np.array(target_quat),
        init_qpos=to_numpy(robot.get_qpos()),
        rot_mask=[False, False, True],
        return_error=True,
    )

    if q is None:
        raise RuntimeError(f"Inverse kinematics failed for end-effector target {target_pos}")

    q_target = to_numpy(q)[:7]
    if not np.all(np.isfinite(q_target)):
        raise RuntimeError(f"IK returned invalid joint positions for target {target_pos}")
    pos_error = np.linalg.norm(to_numpy(ik_error)[:3])
    if pos_error > 0.02:
        print(f"Warning: IK target error is {pos_error:.3f} m; executing the closest pose")
    print(f"IK target {np.asarray(target_pos)}; position error {pos_error:.3f} m")
    q_start = to_numpy(robot.get_qpos())[:7]
    steps = max(40, int(duration / SIM_DT))

    # Interpolate through arm configurations so the controller follows a
    # smooth path rather than jumping directly to a distant IK solution.
    for alpha in np.linspace(0.0, 1.0, steps):
        q_arm = (1.0 - alpha) * q_start + alpha * q_target
        robot.control_dofs_position(
            q_arm,
            dofs_idx_local=ARM_DOFS,
        )
        scene.step()

    for _ in range(12):
        robot.control_dofs_position(q_target, dofs_idx_local=ARM_DOFS)
        scene.step()


def quat_multiply(a, b):
    """Compose two scalar-first (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def rotate_vector(vector, axis, angle):
    """Rotate a world-space vector around a unit axis using Rodrigues' formula."""
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run Genesis on CPU",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI",
    )

    args = parser.parse_args()


    # IMPORTANT:
    # --headless must NOT force CPU.
    backend = gs.cpu if args.cpu else gs.gpu


    # --------------------------------------------------------
    # Genesis initialization
    # --------------------------------------------------------

    gs.init(backend=backend, precision="32")
    print(f"Genesis backend: {'CPU' if args.cpu else 'GPU'} (float32)")


    # --------------------------------------------------------
    # Create temporary URDF
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        delete=False,
    ) as f:

        cup_urdf_path = f.name

    cup_mesh_path = Path(__file__).with_name("water_cup.obj").resolve()
    with open(cup_urdf_path, "w", encoding="utf-8") as f:
        f.write(make_cup_urdf(cup_mesh_path))


    try:

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                # Fewer substeps reduce SPH cost while retaining a conservative
                # About 0.13 ms per SPH substep for 3 mm particles.
                dt=SIM_DT,
                substeps=30,
            ),

            sph_options=gs.options.SPHOptions(
                particle_size=0.003,
                lower_bound=(-0.6, -0.4, 0.0),
                upper_bound=(0.5, 0.4, 0.8),
            ),

            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
                max_dynamic_constraints=8,
            ),

            # TRUE rigid <-> SPH coupling
            coupler_options=gs.options.LegacyCouplerOptions(
                rigid_sph=True,
            ),

            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.9, -1.1, 0.75),
                camera_lookat=(-0.05, 0.0, 0.12),
                camera_fov=40,
            ),

            vis_options=gs.options.VisOptions(
                particle_size_scale=1.0,
            ),

            show_viewer=not args.headless,
        )


        # ----------------------------------------------------
        # Ground
        # ----------------------------------------------------

        scene.add_entity(
            morph=gs.morphs.Plane(),
        )


        # ----------------------------------------------------
        # CUP
        # ----------------------------------------------------

        cup = scene.add_entity(
            morph=gs.morphs.URDF(
                file=cup_urdf_path,

                pos=CUP_POS,

                # Fixed URDF links become one rigid body.
                merge_fixed_links=True,
            ),
            material=gs.materials.Rigid(
                friction=1.5,
                coup_friction=1.0,
                coup_softness=0.008,
                coup_restitution=0.0,
            ),
        )


        # ----------------------------------------------------
        # SPH LIQUID
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We DO NOT move these particles manually.
        #
        # The only forces acting on them come from:
        #   - SPH fluid dynamics
        #   - gravity
        #   - rigid-SHP coupling
        #   - collision with the cup
        #
        # ----------------------------------------------------

        water_material = gs.materials.SPH.Liquid(
            rho=1000.0,
            stiffness=50000.0,
            exponent=7.0,
            # Use the Genesis default water viscosity; keep surface tension
            # low so it does not hold the particles together like a gel.
            mu=0.0001,
            gamma=0.0005,
            sampler="regular",
        )


        water = scene.add_entity(
            material=water_material,

            morph=gs.morphs.Box(
                pos=WATER_CENTER,
                size=WATER_SIZE,
            ),

            # Show each SPH particle as a small blue sphere. Surface
            # reconstruction is expensive and made this small volume appear
            # as a few disconnected blobs.
            surface=gs.surfaces.Default(
                color=(0.04, 0.25, 0.95, 1.0),
                vis_mode="particle",
            ),
        )


        # ----------------------------------------------------
        # FRANKA
        # ----------------------------------------------------

        robot = scene.add_entity(
            morph=gs.morphs.MJCF(
                # Genesis bundles this MJCF asset; use the package asset path
                # supported by Genesis rather than a nonexistent download helper.
                file="xml/franka_emika_panda/panda.xml",
                pos=ROBOT_POS,
            ),
            # The arm never touches the liquid; excluding its many collision
            # geoms from SPH coupling makes the fluid simulation much faster.
            material=gs.materials.Rigid(friction=1.0, needs_coup=False),
        )


        # ----------------------------------------------------
        # Build scene
        # ----------------------------------------------------

        scene.build()

        print(f"Cup collision geoms: {cup.n_geoms}; water particles: {water.n_particles}")


        # ----------------------------------------------------
        # Initial robot configuration
        # ----------------------------------------------------

        initial_arm_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        # Panda's gripper tendon actuator has a very small native gain after
        # MJCF import. Set explicit position gains as in Genesis' Franka demo,
        # otherwise the fingers can stay at their initial closed pose.
        robot.set_dofs_kp(np.array([100.0, 100.0]), FINGER_DOFS)
        robot.set_dofs_kv(np.array([10.0, 10.0]), FINGER_DOFS)
        robot.set_qpos(np.concatenate([initial_arm_q, [GRIP_OPEN, GRIP_OPEN]]))
        robot.control_dofs_position(np.array([GRIP_OPEN, GRIP_OPEN]), FINGER_DOFS)


        # ----------------------------------------------------
        # Let SPH settle
        # ----------------------------------------------------
        #
        # The cup remains stationary.
        # Water is NOT teleported into the cup.
        #
        # ----------------------------------------------------

        print("Settling SPH liquid...")

        for _ in range(40):
            # Position targets are persistent motor commands. Issue them from
            # the first frame so gravity cannot drop the unpowered arm.
            robot.control_dofs_position(initial_arm_q, ARM_DOFS)
            robot.control_dofs_position(np.array([GRIP_OPEN, GRIP_OPEN]), FINGER_DOFS)
            scene.step()

        settled_water = to_numpy(water.get_particles_pos())
        print(
            "Settled water height: "
            f"{settled_water[:, 2].min():.3f}..{settled_water[:, 2].max():.3f} m"
        )

        finger_qpos = to_numpy(robot.get_qpos())[7:9]
        print(f"Gripper opened to {finger_qpos[0]:.3f}, {finger_qpos[1]:.3f} m")
        if np.min(finger_qpos) < 0.03:
            raise RuntimeError(
                "The Panda gripper did not open. Check the finger DOF gains and MJCF controller setup."
            )


        # ----------------------------------------------------
        # Find robot hand
        # ----------------------------------------------------

        hand = robot.get_link("hand")


        # ----------------------------------------------------
        # APPROACH CUP
        # ----------------------------------------------------

        target_quat = np.array([
            0.0,
            1.0,
            0.0,
            0.0,
        ])


        approach_pos = np.array([
            CUP_POS[0],
            CUP_POS[1],
            CUP_POS[2] + 0.05 + HAND_ABOVE_CUP_CENTER + 0.10,
        ])


        print("Moving robot above cup...")

        move_ee(
            scene,
            robot,
            approach_pos,
            target_quat,
            duration=0.7,
        )


        # ----------------------------------------------------
        # LOWER HAND
        # ----------------------------------------------------

        # The Panda fingers close along the hand's local y axis. The cup is
        # square, so a centered side grasp works for either horizontal axis.
        cup_center_z = CUP_POS[2] + 0.05
        grasp_pos = np.array([CUP_POS[0], CUP_POS[1], cup_center_z + HAND_ABOVE_CUP_CENTER])


        print("Moving hand to cup...")

        move_ee(
            scene,
            robot,
            grasp_pos,
            target_quat,
            duration=0.6,
        )


        # ----------------------------------------------------
        # OPEN GRIPPER
        # ----------------------------------------------------

        print("Opening gripper...")
        robot.control_dofs_position(np.array([GRIP_OPEN, GRIP_OPEN]), FINGER_DOFS)
        for _ in range(30):
            scene.step()


        # ----------------------------------------------------
        # CLOSE GRIPPER ON THE CUP WALLS
        # ----------------------------------------------------

        print("Closing gripper...")

        # Close gradually and maintain inward force. The fingers must contact
        # opposite cup walls and support the cup by friction during motion.
        hold_arm_q = to_numpy(robot.get_qpos())[:7].copy()
        for width in np.linspace(GRIP_OPEN, 0.020, 40):
            robot.control_dofs_position(hold_arm_q, ARM_DOFS)
            robot.control_dofs_position(np.array([width, width]), FINGER_DOFS)
            scene.step()
        for _ in range(25):
            robot.control_dofs_position(hold_arm_q, ARM_DOFS)
            robot.control_dofs_force(np.array([GRIP_HOLD_FORCE, GRIP_HOLD_FORCE]), FINGER_DOFS)
            scene.step()


        # ----------------------------------------------------
        # Lift cup
        # ----------------------------------------------------

        lift_pos = np.array([
            CUP_POS[0],
            CUP_POS[1],
            grasp_pos[2] + 0.15,
        ])


        print("Lifting cup...")

        cup_z_before_lift = to_numpy(cup.get_pos())[2]
        move_ee(
            scene,
            robot,
            lift_pos,
            target_quat,
            duration=0.7,
        )

        cup_z_after_lift = to_numpy(cup.get_pos())[2]
        lifted_by = cup_z_after_lift - cup_z_before_lift
        print(f"Cup lift measured: {lifted_by:.3f} m")
        if lifted_by < 0.05:
            robot.control_dofs_position(np.array([GRIP_OPEN, GRIP_OPEN]), FINGER_DOFS)
            for _ in range(30):
                scene.step()
            raise RuntimeError(
                "The fingers did not lift the cup. Check that the hand is centered on the cup walls "
                "and that the gripper closes in the viewer."
            )


        # ----------------------------------------------------
        # Move sideways
        # ----------------------------------------------------

        move_pos = np.array([
            CUP_POS[0] + PLACE_OFFSET[0],
            CUP_POS[1],
            lift_pos[2],
        ])


        print("Moving cup sideways...")

        move_ee(
            scene,
            robot,
            move_pos,
            target_quat,
            duration=0.9,
        )

        # Set the cup down at the destination before tipping it.
        place_pos = np.array([move_pos[0], move_pos[1], grasp_pos[2]])
        print("Lowering cup to the destination...")
        move_ee(scene, robot, place_pos, target_quat, duration=0.6)


        # ----------------------------------------------------
        # Hold the cup on the floor briefly, then tip it so the water spills
        # at the destination. The cup is allowed to fall onto its side.
        # ----------------------------------------------------

        print("Tilting the placed cup to pour the water...")

        # Robot base is at negative X. Tip the cup along the horizontal
        # direction from the robot toward the destination (+X here), so it
        # falls away from the manipulator.
        away_direction = np.asarray(place_pos[:2]) - np.asarray(ROBOT_POS[:2])
        away_direction /= np.linalg.norm(away_direction)
        tilt_axis = np.array([-away_direction[1], away_direction[0], 0.0], dtype=np.float32)
        # Positive rotation around this axis carries the cup's opening along
        # the away direction, toward positive X for the current placement.
        tilt_angle = np.deg2rad(105.0)
        pour_delta = np.array([
            np.cos(tilt_angle / 2),
            *(tilt_axis * np.sin(tilt_angle / 2)),
        ], dtype=np.float32)
        pour_quat = quat_multiply(pour_delta, target_quat)

        # Keep the far lower cup edge fixed on the surface and move the hand
        # along the matching arc. The cup itself is not lifted to pour.
        cup_world_pos = to_numpy(cup.get_pos())
        pivot = cup_world_pos.copy()
        pivot[:2] += away_direction * 0.035  # half of the cup's outer width
        pivot[2] -= 0.004  # bottom surface relative to the cup root
        hand_world_pos = to_numpy(hand.get_pos())
        tipped_hand_pos = pivot + rotate_vector(
            hand_world_pos - pivot,
            tilt_axis,
            tilt_angle,
        )
        move_ee(scene, robot, tipped_hand_pos, pour_quat, duration=0.8)
        pour_arm_q = to_numpy(robot.get_qpos())[:7].copy()

        for _ in range(180):
            robot.control_dofs_position(pour_arm_q, ARM_DOFS)
            robot.control_dofs_force(np.array([GRIP_HOLD_FORCE, GRIP_HOLD_FORCE]), FINGER_DOFS)
            scene.step()


        # ----------------------------------------------------
        # Release the physical finger grasp over the destination.
        # ----------------------------------------------------

        print("Releasing cup...")

        robot.control_dofs_position(np.array([GRIP_OPEN, GRIP_OPEN]), FINGER_DOFS)
        for _ in range(120):
            scene.step()


        # ----------------------------------------------------
        # Move robot away
        # ----------------------------------------------------

        # After the cup has fallen, return the hand upright and lift it clear.
        retreat_pos = place_pos + np.array([0.0, 0.0, 0.10])


        move_ee(
            scene,
            robot,
            retreat_pos,
            target_quat,
            duration=0.6,
        )


        # ----------------------------------------------------
        # Let cup + water fall naturally
        # ----------------------------------------------------

        print("Released. Simulating free motion...")

        for _ in range(60):
            scene.step()


    finally:

        # ----------------------------------------------------
        # Cleanup temporary URDF
        # ----------------------------------------------------

        if os.path.exists(cup_urdf_path):
            os.remove(cup_urdf_path)


if __name__ == "__main__":
    main()
