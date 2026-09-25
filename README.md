# Eye Disease Classification

Deep learning classifier for 8 retinal disease categories (Normal, Diabetic Retinopathy,
Glaucoma, Cataract, AMD, Hypertension, Myopia, Others) from fundus images, built on the
unified 11,839-image dataset aggregated from ODIR-5K, APTOS 2019, ACRIMA and ORIGA.

## Notebooks

- **`Final_CNN_Efficientnet.ipynb`** — main notebook. Scratch CNN, fine-tuned
  EfficientNetB0, and an extra DenseNet121 run, all as single 8-class classifiers.
- **`binary-general.ipynb`** — alternative approach: splits the problem into a screening
  model (Normal vs Disease) followed by a 7-class disease classifier, tried with both
  DenseNet121 and EfficientNetB0 as the backbone.

Both notebooks are self-contained and were run on Colab/Kaggle with a GPU.

## Dataset

The zipped dataset (`EyeDiseaseDataset.zip`) is expected in Google Drive; the first cells
mount Drive, copy the zip, and unzip it. It includes `metadata.csv` (image filenames and
labels), a `splits/val.csv`, a `catalog/` folder with summary stats and reference plots,
and the `images/` folder itself.

## Approach

- **Exploration**: class counts, sample images per class, and a check of image
  dimensions/pixel ranges before choosing a fixed input size (224x224, later 300x300 in
  the binary notebook).
- **Class imbalance** (Normal: 4,698 images vs Hypertension: 88): handled with balanced
  class weights for the scratch CNN, and with oversampling (equal probability per class
  during training) for the pretrained models.
- **Augmentation**: random flip, rotation, zoom, contrast and brightness, applied inside
  the model so it only affects training batches.
- **Models**:
  - Scratch CNN — 4 conv/pool blocks, global average pooling, dense head.
  - EfficientNetB0 / DenseNet121 — ImageNet-pretrained, fine-tuned in two stages (head
    only, then partial/full base unfreeze at a lower learning rate).
- **Evaluation**: per-class precision/recall/F1, plus an accuracy/parameter-count/latency
  comparison table to inform the real-time deployment discussion.

## Results (`Final_CNN_Efficientnet.ipynb`, single 8-class models)

| | Scratch CNN | EfficientNetB0 (fine-tuned) |
|---|---|---|
| Accuracy | 0.34 | 0.63 |
| Macro F1 | 0.25 | 0.59 |
| Weighted F1 | 0.41 | 0.65 |
| Parameters | ~111K | ~4.06M |
| Inference | ~3.2 ms/image | ~5.2 ms/image |

The fine-tuned DenseNet121 run (also in this notebook) reached 0.46 accuracy — lower
than EfficientNetB0 here, mainly because it wasn't given as much fine-tuning as the
EfficientNetB0 run.

Both pretrained models clearly outperform the scratch CNN, especially on minority classes
(Myopia, Cataract) where the scratch CNN struggles to learn anything useful from few
examples. All models are weakest on Hypertension (only 70 training images) and Others
(a catch-all category with no single visual pattern), which is expected given the data.

## Results (`binary-general.ipynb`, screening + disease cascade)

| | DenseNet121 cascade | EfficientNetB0 cascade |
|---|---|---|
| Accuracy | 0.58 | 0.57 |
| Macro F1 | 0.50 | 0.50 |
| Weighted F1 | 0.62 | 0.62 |

Splitting into a screening step (Normal vs Disease) and a disease-only classifier raises
recall substantially on rare classes (e.g. Hypertension recall reaches 0.89 with
DenseNet121) compared to the single 8-class models above, but overall accuracy drops,
since a real disease case that the screening model misses as "Normal" never reaches the
disease classifier at all. This is a genuine precision/recall tradeoff: the cascade is
better if catching rare diseases matters more than overall accuracy, the single 8-class
model is better if overall accuracy is the priority.

## Deployment recommendation

For real-time deployment, EfficientNetB0 is the best balance in this project: roughly
double the scratch CNN's inference time but a very large accuracy gain, and noticeably
faster than deeper/more heavily fine-tuned alternatives while getting the best single-model
accuracy achieved here. The scratch CNN is only preferable if compute is extremely
constrained and the accuracy loss is acceptable. The screening + disease cascade is worth
considering specifically when missing a rare-disease case is costlier than a false alarm,
at the cost of lower overall accuracy and roughly double the inference cost (two models
run per image).

## Requirements

```
tensorflow
numpy
pandas
matplotlib
pillow
scikit-learn
seaborn
```

