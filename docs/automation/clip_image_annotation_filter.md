# Clip image annotation filter

!!! warning
    This feature is just experimental and very low quality.

You can query object from annotated rosbags by prompt.

## Use with CLI

```bash
amber automation clip_image_annotation_filter tests/automation/clip_image_annotation_filter.yaml tests/rosbag/ford_with_annotation/read_images_and_bounding_box.yaml tests/rosbag/ford_with_annotation/bounding_box.mcap output.mcap
```

Task description yaml for the detic_image_labaler is here.

```yaml
target_objects: ["passenger car."] # Target object you want to find.

# If the width of the bounding box overs min_width or the height of the bounding box overs min_height of the bounding box, it recoganize as candidate bounding box.
min_width: 30 # min width of the object in pixels.
min_height: 30 # min height of the object in pixels.
min_area: 50 # min area of the object in pixels.

# Classification method, you can choose from two method.
# You can choose from clip_with_lvis_and_custom_vocabulary, consider_annotation_with_bert
classify_method: consider_annotation_with_bert
# Configuration parameter for consider_annotation_with_bert, this value was used when you choose `consider_annotation_with_bert`
consider_annotation_with_bert_config:
  positive_nagative_ratio: 1.0 # Ratio of the cosine similarity of positive and negative prompt. Negative prompt is "Not a photo of $target_object".
  min_clip_cosine_similarity: 0.25 # Minimum values of cosine similarity with clip text/image embeddings.
  min_clip_cosine_similarity_with_berf: 0.3 # Minimum values of cosine similarity with clip text embeddings and image enbeddings consider prompt similarity using bert.
```

### consider_annotation_with_bert

```python
# Pure clip cosine similarity.
clip_similarity = cosine_similarity(
    clip_embeddings / torch.sum(clip_embeddings),
    self.text_embeddings[target_object][0],
)
# Clip cosine similarity considering bert embeddings.
positive = cosine_similarity(
    clip_embeddings / torch.sum(clip_embeddings)
    + annotation_text_embeddings
    / torch.sum(annotation_text_embeddings)
    * self.text_encoder.cosine_similarity(
        bounding_box.object_class, target_object
    ),
    self.text_embeddings[target_object][0],
)
# Pure clip cosine similarity with negative prompt.
negative = cosine_similarity(
    clip_embeddings / torch.sum(clip_embeddings),
    # - annotation_text_embeddings
    # / torch.sum(annotation_text_embeddings)
    # * bounding_box.score
    # * self.text_encoder.cosine_similarity(
    #     bounding_box.object_class, target_object
    # ),
    self.text_embeddings[target_object][1],
)
```

### clip_with_lvis_and_custom_vocabulary

Embed all object categories in lvis and append custom vocabulary to the text embedding tensor.
Then, find most nearest category.

```python
if self.lvis_text_embeddings == None:
    with torch.no_grad():
        self.lvis_text_embeddings = self.model.encode_text(
            tokenize(self.lvis_prompts).to(self.device)
        )
prompts: List[str] = []
for text in texts:
    prompts.append("A photo of a " + text)
with torch.no_grad():
    text_embeddings = torch.cat(
        [
            self.lvis_text_embeddings,
            self.model.encode_text(tokenize(prompts).to(self.device)),
        ],
        dim=0,
    )
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    similarity = (
        image_embeddings.to(torch.float32) @ text_embeddings.to(torch.float32).T
    ).softmax(dim=-1)
    values, indices = similarity.topk(1)
    for value, index in zip(values, indices):
        if index < len(self.lvis_classes):
            return None
        else:
            return (texts[index - len(self.lvis_classes)], value.item())
```

## Use with Python API

```python
current_path = Path(os.path.dirname(os.path.realpath(__file__)))
filter = ClipImageAnnotationFilter(str(current_path / "automation" / "clip_image_annotation_filter.yaml"))
dataset = ImagesAndAnnotationsDataset(
    str(current_path / "rosbag" / "ford_with_annotation" / "bounding_box.mcap"),
    str(current_path / "rosbag" / "ford_with_annotation" / "read_images_and_bounding_box.yaml"),
)
annotations = filter.inference(dataset)
```
