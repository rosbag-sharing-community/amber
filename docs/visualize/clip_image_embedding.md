# Visualize CLIP Image embedding

## Use with CLI

!!! warning
    This sample command is written with the assumption that it will be executed in the root directory of the amber package.

```bash
amber visualize image_embedding tests/visualization/clip_image_visualization.yaml tests/rosbag/ford_with_annotation/read_images_and_bounding_box.yaml tests/rosbag/ford_with_annotation/bounding_box.mcap
tensorboard --host 0.0.0.0 --port 6006 --logdir runs
```

After executing tensorboard, access [http://0.0.0.0:6006#projector](http://0.0.0.0:6006#projector) by your browser.

In tensorboard, you can see the embedding space of the CLIP.

<iframe width="560" height="315" src="https://www.youtube.com/embed/RbLG4dcH23U?si=DzX4yQdx4n3Qe-aI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Use with Python API

```python
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
visualization = ClipEmbeddingsVisualization(
    str(current_path / "visualization" / "clip_image_visualization.yaml")
)
dataset = ImagesAndAnnotationsDataset(
    str(current_path / "rosbag" / "ford_with_annotation" / "bounding_box.mcap"),
    str(
        current_path
        / "rosbag"
        / "ford_with_annotation"
        / "read_images_and_bounding_box.yaml"
    ),
)
visualization.visualize(dataset)
```
