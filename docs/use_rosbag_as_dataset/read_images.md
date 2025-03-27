# Read images

ImagesDataset Class provides only image data.
Example task description yaml file is here.

```yaml
image_topics:
  - topic_name: /wamv/sensors/cameras/front_left_camera_sensor/image_raw
  - topic_name: /wamv/sensors/cameras/front_right_camera_sensor/image_raw
compressed: true
```

```python
from amber_mcap.dataset.images_and_annotations_dataset import (
    ImagesDataset,
    ReadImagesConfig,
)

dataset = ImagesDataset(
  "(path to rosbag .mcap file)",
  ReadImagesConfig.from_yaml_paht("(path to rosbag yaml description file)"))
```
