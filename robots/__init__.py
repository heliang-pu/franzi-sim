"""The robots, one package each. A robot package keeps everything that is its own:
the source model (URDF and meshes under assets/), the derived simulation scene
and its colliders, the shared FK/IK, the cameras, the appearance, the hardware
built for it and its controller configuration (config/).

So far there is one, franzi, and sim/ and the scenes are written for it. A
second robot gets a package of its own beside it.
"""
