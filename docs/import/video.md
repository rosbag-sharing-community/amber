# Video

Video import tool can convert video files into rosbag.
Currently, only `*.mp4` file types are supported.

## Use with CLI

!!! warning
    This sample command is written with the assumption that it will be executed in the root directory of the amber package.

```bash
amber import video tests/video/soccer_goal.mp4 tests/video_importer.yaml
```

If you want to know the option, please run `amber import video -h` command.

example of the inporter config is here.

```yaml
topic_name: /camera/image_raw
rosbag_path: output.mcap
```

## Use with Python API

```python
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
importer = VideoImporter(
    str(current_path / "video" / "soccer_goal.mp4"),
    str(current_path / "video_importer.yaml"),
)
importer.write()
```
