# train-pi-cam

This is a non-interactive version of the tutorial code published [here](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training).

It builds a Docker container and then runs the training code on the data passed.

## Prerequisites
### Dataset

The dataset should be labelled and exported in the COCO format. There should be
a `result.json` file and an `images` directory in the top-level. The
`result.json` should refer to the pictures without path, just the filename.
Label Studio is convenient to use, just make sure to use a project of the type
"Object Detection with Bounding Boxes".

Here is an example of the hierarchy:

```
$ tree
.
└── my_dataset
   ├── images
   │   ├── cat1.jpg
   │   ├── cat2.jpg
   └── result.json
```

Excerpt from `result.json`:

```
{
  "images": [
    {
      "width": 640,
      "height": 480,
      "id": 0,
      "file_name": "cat1.jpg"
    },
    [...]
  ],
  "categories": [
    {
      "id": 0,
      "name": "cat"
    },
    [...]
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "segmentation": [],
      "bbox": [
        0.0,
        226.0089686098655,
        113.54260089686096,
        126.99551569506725
      ],
      "ignore": 0,
      "iscrowd": 0,
      "area": 14419.401154256064
    },
    [...]
   ]
  }
}
```


### Usage
```
make build
make train NR_EPOCH=100 VAL_INTERVAL=10 DATA_PATH=/path/to/dataset
```

Once the training is complete, the trained model should in `out/packerOut.zip` in the dataset
directory. Send it to the Raspberry Pi, install the package `imx500-tools` and from there run:

```
imx500-package -i packerOut.zip -o packaged-model
```

The `network.rpk` file should get created and is ready to be used.
