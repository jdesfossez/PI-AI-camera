#!/usr/bin/env python3

# # Train NanoDet with custom dataset

# Perform initial checks in order to continue
import shutil
import os
import subprocess
import yaml
import tensorflow as tf
import torch
import sys
from matplotlib import pyplot as plt
from pathlib import Path
import copy
import torch
import numpy as np

from typing import Callable, Iterator, Tuple, List

import cv2
from keras.models import Model

sys.path.insert(0,"./local_mct")
import model_compression_toolkit as mct

from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import coco_dataset_generator, CocoEval
from tutorials.mct_model_garden.models_keras.nanodet.nanodet_keras_model import nanodet_plus_m
from tutorials.mct_model_garden.models_keras.utils.torch2keras_weights_translation import load_state_dict
from tutorials.mct_model_garden.models_keras.nanodet.nanodet_keras_model import nanodet_box_decoding

sys.path.insert(0,"./nanodet")
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, Logger

assert 'local_mct' in mct.__file__, print(mct.__file__)
assert '2.14' in tf.__version__, print(tf.__version__)
assert '1.13' in torch.__version__, print(torch.__version__)

config_path = 'nanodet-plus-m-1.5x_416-ppe.yml'
model_path = '/workspace/nanodet-plus-m-1.5x_416-ppe/model_best/nanodet_model_best.pth'
dst_path = '/workspace/nanodet-plus-m-1.5x_416-ppe/model_best/nanodet_model_best-removed-aux.pth'

# Check shared memory
shm_stats = shutil.disk_usage('/dev/shm')
shm_in_gb = shm_stats.total / (1024 ** 3)
print(f"shm memory: {shm_in_gb:.2f}GB")

print(f'Is cuda available: {torch.cuda.is_available()}')
assert shm_in_gb >= 12 or torch.cuda.is_available()

# Move test/train/valid to dataset folder
DATASET_PATH = '/data'

assert Path(f'{DATASET_PATH}/result.json').exists()

config_file = Path(config_path).read_text()
config_file = config_file.replace("$NR_EPOCH", os.environ["NR_EPOCH"])
config_file = config_file.replace("$VAL_INTERVAL", os.environ["VAL_INTERVAL"])
Path(config_path).write_text(config_file)
config = yaml.safe_load(config_file)
print(config)

#assert config['device']['gpu_ids'] == -1 or config['device']['gpu_ids'] == [0], print(f"gpu_ids: {config['device']['gpu_ids']}")
#assert config['schedule']['total_epochs'] == 2 or config['schedule']['total_epochs'] == 100, print(f"total_epochs: {config['schedule']['total_epochs']}")
#assert config['schedule']['val_intervals'] == 1 or config['schedule']['val_intervals'] == 10, print(f"val_intervals: {config['schedule']['val_intervals']}")


assert '1.13' in torch.__version__, print(torch.__version__)
assert Path(config_path).exists()

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PYTHONPATH"] = f"{dir_path}/nanodet"
subprocess.call(["python3", "nanodet/tools/train.py", config_path])


# # Remove aux layers that are only used during training
def remove_aux(cfg, model_path, remove_layers=['aux_fpn', 'aux_head'], debug=False):
    model = build_model(cfg.model)
    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    if len(remove_layers) > 0:
        state_dict = copy.deepcopy(ckpt['state_dict'])
        for rlayer in remove_layers:
            for layer in ckpt['state_dict']:
                if rlayer in layer:
                    del state_dict[layer]
                    if debug:
                        print(f'removed layer: {layer}')
        del ckpt['state_dict']
        ckpt['state_dict'] = copy.deepcopy(state_dict)
        del state_dict
    return ckpt


load_config(cfg, config_path)
ckpt = remove_aux(cfg, model_path, ['aux_fpn', 'aux_head'])
torch.save(ckpt, dst_path)
print(f'Saved to: {dst_path}')


# # Quantization of custom NanoDet model using Model Compression Toolkit
# Quantization is based on https://github.com/sony/model_optimization/blob/v2.0.0/tutorials/notebooks/keras/ptq/example_keras_nanodet_plus.ipynb

# # Keras NanoDet float model

# Upload the trained custom model
CUSTOM_WEIGHTS_FILE = dst_path  # The NanoDet model trained with PPE dataset
CLASS_NAMES = ['cat', 'dog', 'human']
NUM_CLASSES = len(CLASS_NAMES)

DATASET_TRAIN = '/data/images'
ANNOT_TRAIN = '/data/result.json'
DATASET_VALID = '/data/images'
ANNOT_VALID = '/data/result.json'
DATASET_REPR = DATASET_VALID
ANNOT_REPR = ANNOT_VALID

QUANTIZED_MODEL_NAME = 'nanodet-quant-ppe.keras'

BATCH_SIZE = 2 # BATCH_SIZE * N_ITER should be > number of images
N_ITER = 4  # 1 for testing, otherwise 20

assert Path(CUSTOM_WEIGHTS_FILE).exists()
assert Path(DATASET_REPR).exists()


def get_model(weights=CUSTOM_WEIGHTS_FILE, num_classes=NUM_CLASSES):
    INPUT_RESOLUTION = 416
    INPUT_SHAPE = (INPUT_RESOLUTION, INPUT_RESOLUTION, 3)
    SCALE_FACTOR = 1.5
    BOTTLENECK_RATIO = 0.5
    FEATURE_CHANNELS = 128

    pretrained_weights = torch.load(weights, map_location=torch.device('cpu'))['state_dict']
    # Generate Nanodet base model
    model = nanodet_plus_m(INPUT_SHAPE, SCALE_FACTOR, BOTTLENECK_RATIO, FEATURE_CHANNELS, num_classes)

    # Set the pre-trained weights
    load_state_dict(model, state_dict_torch=pretrained_weights)

    # Add Nanodet Box decoding layer (decode the model outputs to bounding box coordinates)
    scores, boxes = nanodet_box_decoding(model.output, res=INPUT_RESOLUTION, num_classes=num_classes)

    # Add TensorFlow NMS layer
    outputs = tf.image.combined_non_max_suppression(
        boxes,
        scores,
        max_output_size_per_class=300,
        max_total_size=300,
        iou_threshold=0.65,
        score_threshold=0.001,
        pad_per_class=False,
        clip_boxes=False
        )

    model = Model(model.input, outputs, name='Nanodet_plus_m_1.5x_416')

    print('Model is ready for evaluation')
    return model


# known warning:  WARNING: head.distribution_project.project not assigned to keras model !!!
float_model = get_model(CUSTOM_WEIGHTS_FILE, NUM_CLASSES)

def nanodet_preprocess(x):
    img_mean = [103.53, 116.28, 123.675]
    img_std = [57.375, 57.12, 58.395]
    x = cv2.resize(x, (416, 416))
    x = (x - img_mean) / img_std
    return x

def get_representative_dataset(n_iter: int, dataset_loader: Iterator[Tuple]):
    def representative_dataset() -> Iterator[List]:
        ds_iter = iter(dataset_loader)
        for _ in range(n_iter):
            yield [next(ds_iter)[0]]

    return representative_dataset

def quantization(float_model, dataset, annot, n_iter=N_ITER):
    # Load representative dataset
    representative_dataset = coco_dataset_generator(dataset_folder=dataset,
                                                    annotation_file=annot,
                                                    preprocess=nanodet_preprocess,
                                                    batch_size=BATCH_SIZE)

    tpc = mct.get_target_platform_capabilities('tensorflow', 'imx500')

    # Preform post training quantization
    quant_model, _ = mct.ptq.keras_post_training_quantization(
        float_model,
        representative_data_gen=get_representative_dataset(n_iter, representative_dataset),
        target_platform_capabilities=tpc)

    print('Quantized model is ready')
    return quant_model


quant_model = quantization(float_model, DATASET_REPR, ANNOT_REPR)
print(f'Representative dataset: {DATASET_REPR}')


# Observe that loading quantized model might require specification of custom layers,
# see https://github.com/sony/model_optimization/issues/1104
quant_model.save(QUANTIZED_MODEL_NAME)
print(f'Quantized model saved: {QUANTIZED_MODEL_NAME}')


# _todo_: coco evaluation of the custom quantized NanoDet model requires some update to the mct repo.

# # Visualize detection

def load_and_preprocess_image(image_path: str, preprocess: Callable) -> np.ndarray:
    """
    Load and preprocess an image from a given file path.

    Args:
        image_path (str): Path to the image file.
        preprocess (function): Preprocessing function to apply to the loaded image.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = preprocess(image)
    image = np.expand_dims(image, 0)
    return image

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, annotation, scale, class_id, score):
    row = scale[0]
    col = scale[1]
    x_min, y_min = int(annotation[1]*col), int(annotation[0]*row)
    x_max, y_max = int(annotation[3]*col), int(annotation[2]*row)

    color = (0,255,0)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    text = f'{int(class_id)}: {score:.2f}'
    cv2.putText(img, text, (x_min + 10, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# draw all annotation bounding boxes on an image
def annotate_image(img, output, scale, quantized_model=False, threshold=0.55):
    if quantized_model:
        b = output[0].numpy()[0]
        s = output[1].numpy()[0]
        c = output[2].numpy()[0]
    else:
        print('Assuming float model')
        b = output.nmsed_boxes.numpy()[0]
        s = output.nmsed_scores.numpy()[0]
        c = output.nmsed_classes.numpy()[0]
    for index, row in enumerate(b):
        if s[index] > threshold:
            #print(f'row: {row}')
            id = int(c[index])
            draw_bounding_box(img, row, scale, id, s[index])
            print(f'class: {CLASS_NAMES[id]} ({id}), score: {s[index]:.2f}')
    return {'bbox':b, 'score':s, 'classes':c}

# See appendix for results. For 2 epochs, the bounding boxes are not perfect...
# But improves considerably for 20 epochs.
test_img = '/data/images/dog141.jpg'
img = load_and_preprocess_image(f'{test_img}', nanodet_preprocess)
output = quant_model(img)
image = cv2.imread(f'{test_img}')
print(f'image shape: {image.shape}')
r = annotate_image(image, output, scale=image.shape, quantized_model=True)
print(r['score'][0])
#assert r['score'][0] > 0.5, print(f"r['score'][0] > 0.5 failed: {r['score'][0]}")
dst = f'annotated.jpg'
if cv2.imwrite(dst, image):
    print(f'Annotated image saved to: {dst}')
else:
    print(f'Failed saving annotated image')
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

subprocess.call(["imxconv-tf", "-i", QUANTIZED_MODEL_NAME, "-o", "/data/out", "--no-input-persistency"])
