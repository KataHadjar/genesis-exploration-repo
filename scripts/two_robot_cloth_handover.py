import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

import numpy as np
import genesis as gs


ARM_DOFS = np.arange(7)
FINGER_DOFS = np.array([7, 8])
GRIP_OPEN = 0.04
GRIP_CLOSED = 0.0
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
HANDLE_SIZE = 0.05
HANDLE_HALF = 0.5 * HANDLE_SIZE
SEAM_GAP = 0.01
HANDLE_Z = 0.5 * HANDLE_SIZE
CLOTH_X_MIN = -0.24790040 * 0.55
CLOTH_X_MAX = 0.25218286 * 0.55
LEFT_HANDLE_POS = np.array([CLOTH_X_MIN - HANDLE_HALF - SEAM_GAP, 0.0, HANDLE_Z])
RIGHT_HANDLE_POS = np.array([CLOTH_X_MAX + HANDLE_HALF + SEAM_GAP, 0.0, HANDLE_Z])
INITIAL_CLOTH_POS = np.array([0.0, 0.00, 0.01])
PARK_QPOS = np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.04, 0.04])
SECOND_ROBOT_PARK_QPOS = np.array([0.0, -0.65, 0.0, -1.95, 0.0, 1.30, 0.75, 0.04, 0.04])


def set_gripper_open(robot):
    robot.control_dofs_position(np.array([GRIP_OPEN, GRIP_OPEN]), dofs_idx_local=FINGER_DOFS)


def set_gripper_closed(robot):
    robot.control_dofs_position(np.array([GRIP_CLOSED, GRIP_CLOSED]), dofs_idx_local=FINGER_DOFS)


def qpos_np(robot):
    qpos = robot.get_qpos()
    if hasattr(qpos, "detach"):
        return qpos.detach().cpu().numpy()
    return np.asarray(qpos)


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quat_to_rot(quat):
    w, x, y, z = as_numpy(quat)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def fingertip_contact_midpoint(robot):
    # MuJoCo's Panda fingertip collision pads are centered at local z=0.0445 m.
    pad_offset = np.array([0.0, 0.0055, 0.0445])
    pad_centers = []
    for name in ("left_finger", "right_finger"):
        finger = robot.get_link(name)
        finger_pos = as_numpy(finger.get_pos(relative=False))
        finger_rot = quat_to_rot(finger.get_quat(relative=False))
        pad_centers.append(finger_pos + finger_rot @ pad_offset)
    return np.mean(pad_centers, axis=0)


def hand_target_for_grasp_point(robot, hand, target_grasp_pos):
    # Measure the loaded Panda's actual contact point in the hand frame, then
    # rotate that offset into the fixed IK orientation used for every grasp.
    hand_pos = as_numpy(hand.get_pos(relative=False))
    hand_rot = quat_to_rot(hand.get_quat(relative=False))
    grasp_offset_local = hand_rot.T @ (fingertip_contact_midpoint(robot) - hand_pos)
    target_rot = quat_to_rot(GRASP_QUAT)
    return np.array(target_grasp_pos) - target_rot @ grasp_offset_local


def move_ee(scene, robot, ee_link, target_pos, *, gripper_open=True, waypoints=90, settle_steps=35):
    qpos = as_numpy(robot.inverse_kinematics(
        link=ee_link,
        pos=np.array(target_pos),
        quat=GRASP_QUAT,
    ))

    qstart = qpos_np(robot)[:7]
    qtarget = qpos[:7]
    for alpha in np.linspace(0.0, 1.0, waypoints):
        qwaypoint = (1.0 - alpha) * qstart + alpha * qtarget
        robot.control_dofs_position(qwaypoint, dofs_idx_local=ARM_DOFS)
        if gripper_open:
            set_gripper_open(robot)
        else:
            set_gripper_closed(robot)
        scene.step()

    for _ in range(settle_steps):
        robot.control_dofs_position(qpos[:7], dofs_idx_local=ARM_DOFS)
        if gripper_open:
            set_gripper_open(robot)
        else:
            set_gripper_closed(robot)
        scene.step()


def hold_parked(scene, robot, qpos, steps):
    for _ in range(steps):
        robot.control_dofs_position(qpos[:7], dofs_idx_local=ARM_DOFS)
        robot.control_dofs_position(qpos[7:9], dofs_idx_local=FINGER_DOFS)
        scene.step()


def close_gripper(scene, robot, arm_qpos, steps=60):
    for _ in range(steps):
        robot.control_dofs_position(arm_qpos, dofs_idx_local=ARM_DOFS)
        set_gripper_closed(robot)
        scene.step()


def move_ee_while_holding(
    scene,
    moving_robot,
    moving_ee,
    target_pos,
    holder_robot,
    holder_qpos,
    *,
    moving_gripper_open=False,
    waypoints=90,
    settle_steps=35,
):
    qpos = as_numpy(moving_robot.inverse_kinematics(
        link=moving_ee,
        pos=np.array(target_pos),
        quat=GRASP_QUAT,
    ))
    qstart = qpos_np(moving_robot)[:7]
    qtarget = qpos[:7]

    for alpha in np.linspace(0.0, 1.0, waypoints):
        moving_robot.control_dofs_position((1.0 - alpha) * qstart + alpha * qtarget, dofs_idx_local=ARM_DOFS)
        if moving_gripper_open:
            set_gripper_open(moving_robot)
        else:
            set_gripper_closed(moving_robot)
        holder_robot.control_dofs_position(holder_qpos, dofs_idx_local=ARM_DOFS)
        set_gripper_closed(holder_robot)
        scene.step()

    for _ in range(settle_steps):
        moving_robot.control_dofs_position(qtarget, dofs_idx_local=ARM_DOFS)
        if moving_gripper_open:
            set_gripper_open(moving_robot)
        else:
            set_gripper_closed(moving_robot)
        holder_robot.control_dofs_position(holder_qpos, dofs_idx_local=ARM_DOFS)
        set_gripper_closed(holder_robot)
        scene.step()


def move_two_robots_holding(scene, robot1, ee1, target1, robot2, ee2, target2, *, waypoints=120, settle_steps=30):
    q1 = as_numpy(robot1.inverse_kinematics(link=ee1, pos=np.array(target1), quat=GRASP_QUAT))[:7]
    q2 = as_numpy(robot2.inverse_kinematics(link=ee2, pos=np.array(target2), quat=GRASP_QUAT))[:7]
    start1 = qpos_np(robot1)[:7]
    start2 = qpos_np(robot2)[:7]

    for alpha in np.linspace(0.0, 1.0, waypoints):
        robot1.control_dofs_position((1.0 - alpha) * start1 + alpha * q1, dofs_idx_local=ARM_DOFS)
        robot2.control_dofs_position((1.0 - alpha) * start2 + alpha * q2, dofs_idx_local=ARM_DOFS)
        set_gripper_closed(robot1)
        set_gripper_closed(robot2)
        scene.step()

    for _ in range(settle_steps):
        robot1.control_dofs_position(q1, dofs_idx_local=ARM_DOFS)
        robot2.control_dofs_position(q2, dofs_idx_local=ARM_DOFS)
        set_gripper_closed(robot1)
        set_gripper_closed(robot2)
        scene.step()

def main():
    parser = argparse.ArgumentParser(description="Two robots lift and stretch cloth using physically grasped sewn handles.")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show viewer")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="Use CPU backend")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.2, -2.6, 2.1),
            camera_lookat=(0.0, 0.0, 0.55),
            camera_fov=42,
        ),
        sim_options=gs.options.SimOptions(dt=0.01, substeps=5),
        pbd_options=gs.options.PBDOptions(particle_size=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=args.vis,
    )

    coupled_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=1.0)
    scene.add_entity(gs.morphs.Plane(), material=coupled_rigid)

    cloth = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=0.55,
            pos=INITIAL_CLOTH_POS,
            euler=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.PBD.Cloth(static_friction=1.0, kinetic_friction=1.0),
        surface=gs.surfaces.Default(color=(0.85, 0.12, 0.18, 1.0)),
    )

    left_handle = scene.add_entity(
        gs.morphs.Box(size=(HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE), pos=LEFT_HANDLE_POS),
        material=coupled_rigid,
        surface=gs.surfaces.Plastic(color=(0.15, 0.35, 0.85)),
    )
    right_handle = scene.add_entity(
        gs.morphs.Box(size=(HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE), pos=RIGHT_HANDLE_POS),
        material=coupled_rigid,
        surface=gs.surfaces.Plastic(color=(0.15, 0.65, 0.25)),
    )

    robot1 = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(-0.65, 0.0, 0.0),
        ),
        material=coupled_rigid,
    )
    robot2 = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.65, 0.0, 0.0),
            euler=(0, 0, 180),
        ),
        material=coupled_rigid,
    )

    scene.build()

    # Model sewn handle tabs by attaching a small patch of cloth particles to
    # each rigid block. Robot grippers only interact physically with the blocks.
    initial_particles = as_numpy(cloth.get_particles_pos())
    left_edge_x = initial_particles[:, 0].min()
    right_edge_x = initial_particles[:, 0].max()
    left_seam = np.flatnonzero(
        (initial_particles[:, 0] <= left_edge_x + 0.004) & (np.abs(initial_particles[:, 1]) <= 0.015)
    )
    right_seam = np.flatnonzero(
        (initial_particles[:, 0] >= right_edge_x - 0.004) & (np.abs(initial_particles[:, 1]) <= 0.015)
    )
    cloth.fix_particles_to_link(left_handle.base_link.idx, particles_idx_local=left_seam)
    cloth.fix_particles_to_link(right_handle.base_link.idx, particles_idx_local=right_seam)

    ee1 = robot1.get_link("hand")
    ee2 = robot2.get_link("hand")

    for robot in (robot1, robot2):
        robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        robot.set_qpos(PARK_QPOS, zero_velocity=True, skip_forward=True)
        set_gripper_open(robot)

    robot2.set_qpos(SECOND_ROBOT_PARK_QPOS, zero_velocity=True, skip_forward=True)
    set_gripper_open(robot2)
    hold_parked(scene, robot1, PARK_QPOS, 1)
    hold_parked(scene, robot2, SECOND_ROBOT_PARK_QPOS, 1)

    # Robot 1 physically pinches the blue sewn handle and starts lifting its edge.
    left_handle_pos = as_numpy(left_handle.base_link.get_pos(relative=False))
    left_approach = left_handle_pos + np.array([0.0, 0.0, 0.13])
    move_ee(
        scene, robot1, ee1,
        hand_target_for_grasp_point(robot1, ee1, left_approach),
        waypoints=110, settle_steps=25,
    )
    move_ee(
        scene, robot1, ee1,
        hand_target_for_grasp_point(robot1, ee1, left_handle_pos),
        waypoints=55, settle_steps=20,
    )
    robot1_hold_q = qpos_np(robot1)[:7].copy()
    close_gripper(scene, robot1, robot1_hold_q, 70)

    left_lift_pos = left_handle_pos + np.array([0.0, 0.0, 0.16])
    move_ee(
        scene, robot1, ee1,
        hand_target_for_grasp_point(robot1, ee1, left_lift_pos),
        gripper_open=False, waypoints=85, settle_steps=25,
    )
    robot1_hold_q = qpos_np(robot1)[:7].copy()

    # Robot 2 physically grasps the green handle while robot 1 holds the blue one.
    right_handle_pos = as_numpy(right_handle.base_link.get_pos(relative=False))
    right_approach = right_handle_pos + np.array([0.0, 0.0, 0.13])
    move_ee_while_holding(
        scene, robot2, ee2,
        hand_target_for_grasp_point(robot2, ee2, right_approach),
        robot1, robot1_hold_q,
        moving_gripper_open=True, waypoints=120, settle_steps=25,
    )
    move_ee_while_holding(
        scene, robot2, ee2,
        hand_target_for_grasp_point(robot2, ee2, right_handle_pos),
        robot1, robot1_hold_q,
        moving_gripper_open=True, waypoints=55, settle_steps=20,
    )
    robot2_hold_q = qpos_np(robot2)[:7].copy()
    close_gripper(scene, robot2, robot2_hold_q, 70)

    # Robot 1 first stretches, compresses, and pulls the cloth sideways while
    # robot 2 keeps the other sewn handle steady.
    right_hold_q = qpos_np(robot2)[:7].copy()
    left_out = np.array([-0.26, -0.10, 0.27])
    left_in = np.array([-0.065, 0.075, 0.13])
    left_side = np.array([-0.19, 0.12, 0.23])
    for target in (left_out, left_in, left_side, left_lift_pos):
        move_ee_while_holding(
            scene, robot1, ee1,
            hand_target_for_grasp_point(robot1, ee1, target),
            robot2, right_hold_q,
            waypoints=75, settle_steps=18,
        )
        robot1_hold_q = qpos_np(robot1)[:7].copy()

    # Robot 2 takes its turn in the opposite directions while robot 1 holds steady.
    right_hold_pos = as_numpy(right_handle.base_link.get_pos(relative=False))
    robot1_hold_q = qpos_np(robot1)[:7].copy()
    right_out = np.array([0.26, 0.10, 0.27])
    right_in = np.array([0.065, -0.075, 0.13])
    right_side = np.array([0.19, -0.12, 0.23])
    right_return = np.array([right_hold_pos[0], right_hold_pos[1], 0.19])
    for target in (right_out, right_in, right_side, right_return):
        move_ee_while_holding(
            scene, robot2, ee2,
            hand_target_for_grasp_point(robot2, ee2, target),
            robot1, robot1_hold_q,
            waypoints=75, settle_steps=18,
        )
        robot2_hold_q = qpos_np(robot2)[:7].copy()

    # Finally, the robots lift together, pull apart diagonally, and squeeze inward.
    lift_left = np.array([-0.17, -0.02, 0.29])
    lift_right = np.array([0.17, 0.02, 0.29])
    move_two_robots_holding(
        scene, robot1, ee1,
        hand_target_for_grasp_point(robot1, ee1, lift_left),
        robot2, ee2,
        hand_target_for_grasp_point(robot2, ee2, lift_right),
        waypoints=100, settle_steps=25,
    )

    stretch_left = np.array([-0.24, -0.08, 0.32])
    stretch_right = np.array([0.24, 0.08, 0.32])
    move_two_robots_holding(
        scene, robot1, ee1,
        hand_target_for_grasp_point(robot1, ee1, stretch_left),
        robot2, ee2,
        hand_target_for_grasp_point(robot2, ee2, stretch_right),
        waypoints=110, settle_steps=25,
    )

    squeeze_left = np.array([-0.07, 0.035, 0.18])
    squeeze_right = np.array([0.07, -0.035, 0.18])
    move_two_robots_holding(
        scene, robot1, ee1,
        hand_target_for_grasp_point(robot1, ee1, squeeze_left),
        robot2, ee2,
        hand_target_for_grasp_point(robot2, ee2, squeeze_right),
        waypoints=100, settle_steps=25,
    )

    move_two_robots_holding(
        scene, robot1, ee1,
        hand_target_for_grasp_point(robot1, ee1, stretch_left),
        robot2, ee2,
        hand_target_for_grasp_point(robot2, ee2, stretch_right),
        waypoints=100, settle_steps=30,
    )

    for _ in range(180):
        robot1.control_dofs_position(qpos_np(robot1)[:7], dofs_idx_local=ARM_DOFS)
        robot2.control_dofs_position(qpos_np(robot2)[:7], dofs_idx_local=ARM_DOFS)
        set_gripper_closed(robot1)
        set_gripper_closed(robot2)
        scene.step()


if __name__ == "__main__":
    main()
