# Read Images

In image_only task, Rosbag2Dataset Class provides only image data.
Example task description yaml file is here.

```yaml
dataset_type: image_only
image_topics:
  - topic_name: /wamv/sensors/cameras/front_left_camera_sensor/image_raw
    compressed: true
  - topic_name: /wamv/sensors/cameras/front_right_camera_sensor/image_raw
```
