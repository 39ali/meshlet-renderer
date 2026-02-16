
Meshlet renderer experiment using WebGPU for rendering millions of triangles.

### Features
- Frustum culling
- Backface culling
- 2 pass Occlusion culling using a hierarchical Z-buffer
- GPU-based rendering

### Things to Improve
- Add meshlet LOD.
- Add mesh instance culling. Instead of culling meshlets first, attempt to cull the mesh instance. This could potentially save time by skipping meshlet culling if the mesh is culled.
- Since WebGPU doesn't support MultiDrawIndirect, the indirect draw vertex count is set to the maximum number of vertices per meshlet (`maxTriPerMeshlet * 3`). This wastes some GPU time.




## building
you can run it natively by building using cmake or on [web](https://39ali.github.io/meshlet-renderer/)

### controls
hold shift and use mouse + WASD for movement
