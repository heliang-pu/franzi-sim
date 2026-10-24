"""What every simulation scene shares: the derived robot scene with its table
and lights (build_task, appearance, fittings, gripper_linkage, wrist_mount),
the policy cameras (camera_rig) and the episode framework (episode.Scene). The
scenes themselves, each with its layout, expert, acceptance and assets, live in
scenes/.

Run the scripts from the project root as modules, e.g.
``python -m sim.build_task``. The shared kinematics, robot_kinematics.py, stays
at the top level.
"""
