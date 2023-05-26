# Detic image labaler

[Detic](https://github.com/facebookresearch/Detic) is a deep learning algorithum developed by facebook research.
This tool generate annotation data by using detic.
Detic can classify 21k classes.

## Use with CLI

!!! warning
    This sample command is written with the assumption that it will be executed in the root directory of the amber package.

```bash
amber automation tests/detic_image_labeler.yaml tests/read_image_ford.yaml tests/rosbag/ford/ford.mcap
```

Task description yaml for the detic_image_labaler is here.

```yaml
task_type: detic_image_labeler # This line should be detic_image_labeler
confidence_threshold: 0.5      # If the confidence overs the threshold, detic determines the object are exists.
video_output_path: output.mp4  # Relative path to the visualization result.
docker_config:                 # Docker configuration
  use_gpu: false               # If true, run with CUDA
```

After executing this command, `output.mp4` movie file was generated.
<iframe width="560" height="315" src="https://www.youtube.com/embed/OPR3ZVzRXCM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Use with Python API

```python
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
labeler = DeticImageLabeler(str(current_path / "detic_image_labeler.yaml"))
dataset = Rosbag2Dataset(
    str(current_path / "rosbag" / "ford" / "ford.mcap"),
    str(current_path / "read_image_ford.yaml"),
)
labeler.inference(dataset)
```

`detic_image_labeler.yaml` and `read_image_ford.yaml` are exactly same when you use detic_image_labaler with CLI.

After executing this command, `output.mp4` movie file was generated.
<iframe width="560" height="315" src="https://www.youtube.com/embed/OPR3ZVzRXCM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
