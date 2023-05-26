# Automation

Automation helps you to enjoy your ML life.
Automation tools can be used with two ways, use with CLI and use with Python API

## CLI tools

All of the automation tools can be use `amber automation` command.

```bash
usage: amber automation [-h] task_description_yaml_path dataset_description_yaml_path rosbag_path

positional arguments:
  task_description_yaml_path
                        Path to the yaml description file path
  dataset_description_yaml_path
                        Path to the yaml description file path for dataset
  rosbag_path           Path to the target rosbag path

options:
  -h, --help            show this help message and exit
```

## Python API

All of the automation tools have Python classes.
If you want to know detail, please read detatiled documentations.

## Support status

| Name                                        | Docker Support     | CUDA Support(Docker) | Native Support | CUDA Support(Native) | Huggingface Support |
|---------------------------------------------|--------------------|----------------------|----------------|----------------------|---------------------|
| [DeticImageLabaler](../detic_image_labaler) | :heavy_check_mark: | :heavy_check_mark:   |                |                      |                     |

### Docker Support
Support automation algorithm inside docker.

### Native Support
Support automation algorithm in native environment.

### Hugging face Support
Support running automation algorithum on [hugging face.](https://huggingface.co/)

### CUDA Support
Support cuda for accelerating automation.
