# Detic image labaler

[Detic](https://github.com/facebookresearch/Detic) is a deep learning algorithum developed by facebook research.
This tool generate annotation data by using detic.
Detic can classify 21k classes.
This tools are running onnx converted detic models with opset=16 in [this repository](https://github.com/axinc-ai/ailia-models/tree/master/object_detection/detic).
Thank you for ailia-models developers.

## Use with CLI

!!! warning
    This sample command is written with the assumption that it will be executed in the root directory of the amber package.

```bash
amber automation detic_image_labeler tests/automation/detic_image_labeler.yaml tests/rosbag/ford/read_image.yaml tests/rosbag/ford/ford.mcap output.mcap
```

Task description yaml for the detic_image_labaler is here.

```yaml
confidence_threshold: 0.5      # If the confidence overs the threshold, detic determines the object are exists.
video_output_path: output.mp4  # Relative path to the visualization result.
vocabulary: "lvis"             # Vocabulary of detic, you can choose from "lvis" and "imagenet_21k"
model_type: "SwinB_896_4x"     # Model type of detic, you can choose from "SwinB_896_4x" and "R50_640_4x"
```

After executing this command, `output.mp4` movie file was generated.
<iframe width="560" height="315" src="https://www.youtube.com/embed/OPR3ZVzRXCM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Use with Python API

```python
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
labeler = DeticImageLabeler(str(current_path / "automation" / "detic_image_labeler.yaml"))
dataset = ImagesDataset(
    str(current_path / "rosbag" / "ford" / "ford.mcap"),
    str(current_path / "rosbag" / "ford" / "read_image.yaml"),
)
labeler.inference(dataset)
```

`detic_image_labeler.yaml` and `read_image.yaml` are exactly same when you use detic_image_labaler with CLI.

After executing this command, `output.mp4` movie file was generated.
<iframe width="560" height="315" src="https://www.youtube.com/embed/OPR3ZVzRXCM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
