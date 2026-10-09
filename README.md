# Rice Leaf Disease Classification

- This project classifies rice leaf photos into three disease types using convolutional neural networks built with TensorFlow/Keras. It was done as part of my data science internship project at DataMites (PRCP-1001), and it is my first project working with image data instead of structured data.

## Problem Statement

- Rice leaf diseases are normally identified by farmers or experts looking at the plant, which takes time and needs experience. The goal here is to build a model that can look at a photo of a rice leaf and predict the disease, so problems can be caught early before they hurt the yield.

## Dataset

- The dataset has 119 jpg images across three classes: Bacterial leaf blight (40), Brown spot (40), and Leaf smut (39). It is almost perfectly balanced, but very small for deep learning. The photos are also not consistent in style. Some are wide full-leaf shots (for example 897x3081 pixels), while others are tight close-ups of just the diseased patch, so image sizes vary a lot.

## What I did

- Loaded the images straight from the class folders with `image_dataset_from_directory` and checked that all 119 images and 3 classes were picked up.
- Looked at sample images, the class counts, and the original image sizes, which showed that photos were not captured in a consistent size or framing.
- Resized every image to 160x160 and made a stratified 70/15/15 train, validation, and test split.
- Fixed the random seed so results are repeatable.
- Used the validation set to pick the best epoch and kept the test set for final reporting only, because using the same images for both inflates the reported accuracy.
- Trained a baseline CNN from scratch (3 Conv + MaxPooling blocks, a dense layer, and Dropout 0.4).
- Trained the same CNN with data augmentation (random flip, rotation, and zoom), so any change in results comes from the augmentation and not from a different architecture.
- Applied transfer learning with a pretrained MobileNetV2 (frozen base, with a small classifier head on top).
- Fine-tuned the last 20 layers of MobileNetV2 with a very low learning rate (1e-5) to see if it would improve further.

## Results

| Model | Test Accuracy |
|---|---|
| Baseline CNN | 65.2% |
| CNN with Data Augmentation | 78.3% |
| Transfer Learning (MobileNetV2, frozen) | 87.0% |
| Fine-tuned Transfer Learning | 82.6% |

- Transfer Learning with a frozen MobileNetV2 base gave the best performance, so it was picked as the final model. The baseline CNN overfit badly, with training accuracy climbing while validation accuracy stayed unstable. Augmentation improved the result noticeably, and transfer learning improved it further. Fine-tuning did not help and actually lowered accuracy, because unfreezing more layers gave the model more room to overfit on such a small training set.

- The test set is very small, so each wrong prediction moves accuracy by several percentage points. The results should be read as a general trend, not as exact numbers.

## Key takeaway

- With only about 40 images per class, a CNN trained from scratch cannot learn strong features, which is why the pretrained MobileNetV2 did so much better. It already knows general visual patterns from over a million images, so only a small classifier had to be trained. More flexibility is not always better either, since fine-tuning more layers hurt performance on a dataset this small.

## Tools used

- Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## Files

- `RiceLeaf_disease.ipynb` - full notebook with data loading, splitting, modeling, comparison report, and challenges report
- `best_model.keras` - saved baseline CNN
- `best_aug_model.keras` - saved augmented CNN
- `final_transfer_model.keras` - saved transfer learning model

- The dataset is not included in the repo. Extract the three class zips into the same folder as the notebook before running it.
