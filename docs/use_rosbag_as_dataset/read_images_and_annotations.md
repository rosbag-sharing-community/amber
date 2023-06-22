# Read images and annotations

ImagesAndAnnotationsDataset Class provides image and annotations.
Example task description yaml file is here.

```yaml
image_topics:
  - topic_name: /image_front_left
annotation_topic: /detic_image_labeler/annotation
compressed: false
```

```python
dataset = ImagesAndAnnotationsDataset("(path to rosbag .mcap file)", "(path to rosbag yaml description file)")
```

## Supported annotation type

| Name         | Support status     | Remarks |
|--------------|--------------------|---------|
| Bounding Box | :heavy_check_mark: |         |
| Polygon      |                    |         |
| Mask         |                    |         |
