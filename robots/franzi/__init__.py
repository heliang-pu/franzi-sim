"""The Inovance/Franzi wheeled dual-arm robot (wheel_robot_26.8.16_3; docs/robot_model.md).

- kinematics: the one shared FK/IK (Pinocchio on the source URDF), the joint
  limits and the SDK joint mapping. Needs no physics engine.
- scene: the derived robot scene, ``build()``: the robot from model (URDF to
  MJCF) with its colliders (link_hulls), the gripper linkage (gripper_linkage)
  and the appearance (appearance, fittings), on the plain table under the lights.
  ``python -m robots.franzi.scene`` writes outputs/robot_scene.xml.
- camera_rig: the policy cameras; wrist_mount: the wrist D405 adapter, whose
  printed part is hardware/wrist_d405_mount/.
- config/: the controller limits, the SDK <-> URDF mapping, the tool points and
  the planning constraints, which simbridge reads too.
- assets/urdf/: the source project's unmodified export (never edit it);
  assets/appearance/ and assets/derived/ are the caches built from it.
"""
