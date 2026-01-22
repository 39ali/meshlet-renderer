implemenation of [gaussian splatting paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), this also implements gpu radix sort for splats sorting in WebGpu for realtime rendering, gaussians are packed to 40bytes since we only need them for rendering and not ML. 


## building 
you can run it natively by building using cmake or on [web](https://39ali.github.io/gaussian-splatting/)

### controls
hold shift and use mouse + wasd for movement
