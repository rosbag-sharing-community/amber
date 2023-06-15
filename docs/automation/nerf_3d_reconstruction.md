# NeRF 3D Reconstruction

NeRF is a deep learning rendering algorithm called Neural radiance fields.
AMBER works with [nerf-studio](https://github.com/nerfstudio-project/nerfstudio), where various NeRF algorithms are implemented, to learn NeRF from image data recorded in rosbag.

Special thanks to nerfstudio contributers.

!!! warning
    Limitation : NeRF 3D reconstruction feature is just developed and not well-used.
    This feature currently supports nerfacto and poisson surface reconstruction method.

## Use with CLI

!!! warning
    Currently, generated mesh and pointcloud cannot be write into rosbag. So, final argument `hoge.mcap` does not work.

```bash
amber automation nerf tests/nerf_3d_reconstruction.yaml tests/rosbag/soccer_goal/read_image.yaml tests/rosbag/soccer_goal/ hoge.mcap
```

## Use with Python API

```python
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
labeler = Nerf3DReconstruction(str(current_path / "automation" / "nerf_3d_reconstruction.yaml"))
dataset = ImagesDataset(
    str(current_path / "rosbag" / "soccer_goal"),
    str(current_path / "rosbag" / "soccer_goal", "read_image.yaml"),
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
│   Config File          │ /workspace/outputs/checkpoints/camera_poses/nerfacto/2023-06-10_100233/config.yml          │
│   Checkpoint Directory │ /workspace/outputs/checkpoints/camera_poses/nerfacto/2023-06-10_100233/nerfstudio_models   │
│                        ╵                                                                                            │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Printing profiling stats, from longest to shortest duration in seconds

VanillaPipeline.get_average_eval_image_metrics: 11.2157

VanillaPipeline.get_eval_image_metrics_and_images: 0.4561

VanillaPipeline.get_eval_loss_dict: 0.0392

Trainer.train_iteration: 0.0260

VanillaPipeline.get_train_loss_dict: 0.0170

Trainer.eval_iteration: 0.0014

find: paths must precede expression: `|'

find: paths must precede expression: `|'

Successfully copied 420.6MB to /tmp/nerf_3d_reconstruction/6d1873fd-fa94-4c1a-a592-b4fd5cbc927e/outputs
Artifacts are outputed under /tmp/nerf_3d_reconstruction/6d1873fd-fa94-4c1a-a592-b4fd5cbc927e/outputs
If you want to check the trained result, please type commands below :
docker run -it --rm --gpus all -p 7007:7007             -v /tmp/nerf_3d_reconstruction/6d1873fd-fa94-4c1a-a592-b4fd5cbc927e:/workspace dromni/nerfstudio:0.3.1/bin/bash -c "find -name config.yml | xargs -I {} ns-viewer --load-config {}"
```

If you want to see the NeRF result, please type final commands shown in the result like below.　　

```bash
docker run -it --rm --gpus all -p 7007:7007             -v /tmp/nerf_3d_reconstruction/6d1873fd-fa94-4c1a-a592-b4fd5cbc927e:/workspace dromni/nerfstudio:0.3.1/bin/bash -c "find -name config.yml | xargs -I {} ns-viewer --load-config {}"
```

Open url showed in console.
Then, you can see result like below.

<iframe width="560" height="315" src="https://www.youtube.com/embed/NgEIB4TRRTo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
