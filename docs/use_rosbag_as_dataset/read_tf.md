# Read tf

ImagesDataset Class provides only transform data.
Example task description yaml file is here.

```yaml
topic_name: "/tf"
static_tf_topic_name: "/tf_static"
target_frame: body
source_frame: map
```

```python
from amber.dataset.tf_dataset import TfDataset, ReadTfTopicConfig

current_path = Path(os.path.dirname(os.path.realpath(__file__)))
dataset = TfDataset(
    str(current_path / "rosbag" / "ford" / "ford.mcap"),
    ReadTfTopicConfig.from_yaml_file(
        str(current_path / "rosbag" / "ford" / "read_tf.yaml")
    ),
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
count = 0
for i_batch, sample_batched in enumerate(dataloader):
    for sample in sample_batched:
        count = count + 1
```
