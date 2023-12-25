# Image Search

This application compares the image embedding calculated from the images in the rosbag with the text embedding calculated from the prompts entered by the user using various Vision&Language models, and presents the closest result as the search result.

This application uses [qdrant](https://qdrant.tech/), a type of vector search engine, for searching.

```plantuml format="svg"
@startuml

left to right direction

database qdrant
actor user

folder rosbag_directory {
  file rosbag.mcap
  file dataset.yaml
}

artifact rosbag2dataset

agent image_encoder

user --> rosbag_directory : 1-1.specify rosbag_direcotry

rosbag.mcap --> rosbag2dataset : 1-2.load images inside rosbag
rosbag2dataset --> image_encoder : 1-3.image embeddings

image_encoder -left-> qdrant : 1-4.upsert image embeddings with metadata

interface gradio_UI

user --> gradio_UI : 2-1.send prompt

agent text_encoder

gradio_UI --> text_encoder : 2-2.calculate text embedding from prompt.

text_encoder --> qdrant : 2-3.query image embedding by text embedding.

qdrant -right-> gradio_UI : 2-4.send search result

gradio_UI --> user : 2.5.see search result

@enduml
```

1-* shows the processing steps up to the point where the embedding of the image is calculated and it is registered in qdrant.
2-* indicates the point where prompt input is accepted and displayed.

## Run application with ford dataset.

!!! warning
    This sample command is written with the assumption that it will be executed in the root directory of the amber package.

```bash
python3 amber/apps/image_search.py --rosbag_directory tests/rosbag/ford/ --sampling_duration=0.1
```
