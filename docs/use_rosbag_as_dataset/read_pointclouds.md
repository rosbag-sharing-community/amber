# Read images

PointcloudDataset Class provides only image data.
Example task description yaml file is here.

```yaml
pointcloud_topics:
  - topic_name: /wamv/sensors/lidars/lidar_wamv_sensor/points
compressed: false
```

```python
from amber_mcap.dataset.pointcloud_dataset import PointcloudDataset, ReadPointCloudConfig

dataset = PointcloudDataset(
  "(path to rosbag .mcap file)",
  ReadPointCloudConfig.from_yaml_file("(path to rosbag yaml description file)"))
```
