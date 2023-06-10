# NeRF 3D Reconstruction

NeRF is a deep learning rendering algorithm called Neural radiance fields.
AMBER works with [nerf-studio](https://github.com/nerfstudio-project/nerfstudio), where various NeRF algorithms are implemented, to learn NeRF from image data recorded in rosbag.

Special thanks to nerfstudio contributers.

## Use with CLI

!!! warning
    Currently, generated mesh and pointcloud cannot be write into rosbag. So, final argument `hoge.mcap` does not work.

```bash
amber automation nerf tests/nerf_3d_reconstruction.yaml tests/read_images_soccer_goal.yaml tests/rosbag/soccer_goal/ hoge.mcap
```

## Use with Python API

```python
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
labeler = Nerf3DReconstruction(str(current_path / "nerf_3d_reconstruction.yaml"))
dataset = Rosbag2Dataset(
    str(current_path / "rosbag" / "soccer_goal"),
    str(current_path / "read_images_soccer_goal.yaml"),
)
labeler.inference(dataset)
```

`nerf_3d_reconstruction.yaml` and `read_image_ford.yaml` are exactly same when you use detic_image_labaler with CLI.

## How to see the training result.

When the training and exporting 3D mesh finished.
Output like below was generated.

```bash
╭────────────────────────────────────────────── 🎉 Training Finished 🎉 ──────────────────────────────────────────────╮
│                        ╷                                                                                            │
│   Config File          │ /workspace/outputs/checkpoints/camera_poses/nerfacto/2023-06-10_091146/config.yml          │
│   Checkpoint Directory │ /workspace/outputs/checkpoints/camera_poses/nerfacto/2023-06-10_091146/nerfstudio_models   │
│                        ╵                                                                                            │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Printing profiling stats, from longest to shortest duration in seconds

VanillaPipeline.get_average_eval_image_metrics: 11.2313

VanillaPipeline.get_eval_image_metrics_and_images: 0.4564

VanillaPipeline.get_eval_loss_dict: 0.0393

Trainer.train_iteration: 0.0262

VanillaPipeline.get_train_loss_dict: 0.0171

Trainer.eval_iteration: 0.0014

find: paths must precede expression: `|'

find: paths must precede expression: `|'

Successfully copied 430.2MB to /tmp/nerf_3d_reconstruction/040a4ab1-8c60-42d3-81f0-6b7214cd5899/outputs
Artifacts are outputed under /tmp/nerf_3d_reconstruction/040a4ab1-8c60-42d3-81f0-6b7214cd5899/outputs
```

If you want to see the NeRF result, please type final commands shown in the result like below.　　

Then, you can see result like below.

<iframe width="560" height="315" src="https://www.youtube.com/embed/NgEIB4TRRTo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
