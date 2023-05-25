# Detic image labaler

!!! warning
    This sample command is written with the assumption that it will be executed in the root directory of the amber package.

```bash
amber automation tests/detic_image_labeler.yaml tests/read_image_ford.yaml tests/rosbag/ford/ford.mcap
```

After executing this command, `output.mp4` movie file was generated.

<iframe width="560" height="315" src="https://www.youtube.com/embed/OPR3ZVzRXCM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

[Detic](https://github.com/facebookresearch/Detic) is a deep learning algorithum developed by facebook research.
This tool generate annotation data by using detic.
Detic can classify 21k classes.
