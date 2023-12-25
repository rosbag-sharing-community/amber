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

If it works correctly, the following message is displayed.

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

Please access [this URL](http://127.0.0.1:7860) as soon as the message is confirmed.

You can search for images by entering a prompt in gradio's UI as shown in the video below.

<iframe width="560" height="315" src="https://www.youtube.com/embed/ryp29wm46TQ?si=ZDRpYwQRO09ogdMZ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
