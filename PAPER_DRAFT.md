# NeuroVision AI: A Research Prototype for Binary Brain-Image Classification and Interactive Visualization

> **Scope and evidence note:** This manuscript describes the software and one exploratory evaluation of the local checkpoint. The checkpoint's training provenance and the image dataset's source and patient-level independence have not been verified. The reported image-level metrics are descriptive, must not be interpreted as clinical performance, and should not be submitted as a validated diagnostic study.

## Abstract

### Background

Brain-image classification is an active area of medical-imaging research. Research prototypes can combine image-classification models with visualization and reporting interfaces, but software functionality alone does not establish clinical validity.

### Objective

This work describes a browser-based prototype for binary classification of brain images into tumor and no-tumor classes, with supplementary model-activation visualizations and an interactive interface. We characterize the implemented workflow and report an exploratory evaluation of the locally available checkpoint on the local test-folder images. The aim is descriptive software characterization, not clinical validation.

### Methods

The prototype uses a ResNet-18 image-classification architecture initialized with ImageNet-pretrained weights and a replacement two-class output layer. Images are resized to 224 × 224 pixels and normalized using ImageNet channel statistics. The training script applies random horizontal flips, rotations up to 10 degrees, and brightness/contrast jitter to training images. It uses cross-entropy loss, the Adam optimizer (learning rate 0.001), batch size 32, and 10 training epochs. The preprocessing script generates an image-level 80:20 train/test split with `random_state=42`; patient-level grouping and stratification are not implemented. We loaded the local checkpoint and evaluated it once on 920 images in the local test folder. This split and checkpoint are not independently verified as an untouched patient-level test set.

### Results

On the local test-folder set (399 no-tumor and 521 tumor images), the loaded checkpoint predicted every image as tumor. The image-level confusion matrix contained 0 true negatives, 399 false positives, 0 false negatives, and 521 true positives. Accuracy was 56.63% (Wilson 95% interval 53.41–59.80%), tumor sensitivity 100%, no-tumor specificity 0%, tumor precision 56.63%, tumor F1 72.31%, balanced accuracy 50.00%, and ROC-AUC 0.487. These descriptive results show degenerate all-positive behavior and are not evidence of useful classification. The test split's patient independence and checkpoint provenance are unknown.

### Conclusions

The repository demonstrates an integrated software workflow for binary image classification and interactive visualization. The current implementation is a research/educational prototype. Dataset provenance, patient-independent evaluation, external validation, and clinical utility have not been established by the software itself.

**Keywords:** brain imaging; image classification; ResNet-18; Grad-CAM; explainable artificial intelligence; research prototype

---

## 1. Introduction

Automated analysis of brain images is a broad research area with potential applications in image triage, computer-assisted review, and methodological investigation. Any such system must be assessed in relation to the provenance and representativeness of its data, the independence of its training and evaluation cohorts, the validity of its labels, and the intended clinical setting.

This work describes NeuroVision AI, a software prototype that combines a binary image classifier with an interactive web interface. The prototype accepts a single two-dimensional PNG or JPEG image, applies image preprocessing, returns scores for two classes, and presents supplementary visualizations and reporting features. The implementation is intended to support research and educational exploration rather than clinical diagnosis.

The project should be described narrowly: its trained task is tumor-versus-no-tumor image classification. It does not perform tumor-subtype classification, volumetric segmentation, or validated anatomical localization. The user interface contains features with broader clinical labels, but these should not be interpreted as validated model capabilities.

### 1.1 Study objective

The objective was to describe the software implementation and characterize the local checkpoint's behavior on the available image-level test folder. The evaluation was exploratory and was not designed to establish clinical performance or generalization.

### 1.2 Research questions

The work addresses two descriptive questions: (1) what image-classification and visualization workflow is implemented, and (2) how did the available checkpoint classify the local test-folder images? It does not evaluate patient-independent performance, localization accuracy against annotations, usability, or clinical benefit.

### 1.3 Contributions

The implemented software contributions are:

- a Streamlit interface for uploading and classifying a single two-dimensional image;
- a binary ResNet-18 inference workflow with softmax score display;
- Grad-CAM and intermediate feature-map visualizations;
- a local, rule-based information panel and PDF report generation;
- optional remote language-model chat/image description and browser-based speech synthesis.

These are implementation contributions, not evidence of diagnostic accuracy, improved clinical outcomes, or validated accessibility benefit.

## 2. Materials and Methods

### 2.1 Software and architecture

The application is implemented in Python. The user interface is built with Streamlit. Image classification uses PyTorch and Torchvision, image handling uses Pillow and OpenCV, and plots use Plotly. PDF reports are generated using FPDF. Optional generative-AI requests use the OpenAI-compatible Python client pointed at the OpenRouter API.

The system consists of the following components:

1. **User interface:** `BrainTumorAI/app.py` manages navigation, upload controls, inference, plots, session state, and report/download interactions.
2. **Preprocessing script:** `BrainTumorAI/preprocess.py` reads the expected source folders, resizes readable images, assigns binary folder labels, and writes train/test directories.
3. **Training script:** `BrainTumorAI/train_classifier.py` defines the augmentation pipeline, loads folder-based datasets, fine-tunes ResNet-18, and saves a model state dictionary.
4. **Explainability visualization:** `BrainTumorAI/heatmap_gen.py` computes a Grad-CAM heatmap using forward activations and backward gradients from a selected layer.
5. **Three-dimensional illustration:** `BrainTumorAI/surgery_viz.py` constructs a point-cloud-style Plotly visualization from a two-dimensional image. It does not reconstruct a scan volume.
6. **Reports and fixed informational text:** `BrainTumorAI/medication_report.py` provides predefined text and PDF construction.
7. **Optional generative AI:** `BrainTumorAI/gemini_assistant.py` sends chat prompts or an image plus prompt to a remote model when configured.

The trained checkpoint is saved locally as `BrainTumorAI/models/tumor_classifier.pth`. Checkpoints (`*.pth`) and datasets are excluded by the repository's Git ignore rules. Consequently, the checkpoint and data used in a local run must be separately archived and identified to reproduce that run.

### 2.2 Dataset and labels

The preprocessing code expects two source directories named `Brain Tumor` and `Healthy` under `archive/Brain Tumor Data Set/Brain Tumor Data Set/`. It maps these folders to `tumor` and `no_tumor`, respectively.

At the time the project documentation was prepared, the local generated image folders contained:

| Split | No tumor | Tumor | Total |
|---|---:|---:|---:|
| Training | 1,688 | 1,992 | 3,680 |
| Test | 399 | 521 | 920 |
| Total | 2,087 | 2,513 | 4,600 |

These are local file counts and are not independently verified cohort counts. The repository does not identify the dataset's authoritative source, release, license, acquisition protocol, patient identities, label adjudication process, or exclusion criteria. **Do not report these counts as a study cohort until the source data and processing history have been checked.**

The preprocessing code uses `train_test_split` with `test_size=0.2` and `random_state=42`, but does not set stratification or group by patient. It does not check for duplicate images. If images include multiple slices from one patient, this split may place related images in both sets. A patient-level split should be created before using the test set to support a scientific performance claim.

### 2.3 Image preprocessing

For dataset preparation, OpenCV reads images and resizes them to 224 × 224 pixels. Files that cannot be read and subdirectories are skipped. The resulting arrays are saved as JPEG files in class-specific `train` and `test` folders.

For model training, Torchvision transforms resize images to 224 × 224, convert them to tensors, and normalize channels with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Training-only augmentation consists of random horizontal flips, random rotations up to 10 degrees, and brightness/contrast jitter with magnitude 0.2.

At dashboard inference, an uploaded PNG, JPG, or JPEG is converted to RGB. Optional brightness, contrast, and sharpness adjustments are applied before resizing and normalization. These controls change the input presented to the classifier and should be held fixed or explicitly recorded in a controlled evaluation.

### 2.4 Model

The training script initializes `torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`, replaces its final fully connected layer with a linear layer containing two outputs, and trains the model parameters [1,2]. The intended class index mapping is `no_tumor: 0`, `tumor: 1`, corresponding to alphabetical ordering by Torchvision `ImageFolder`.

Training configuration in the current script:

| Parameter | Value |
|---|---|
| Backbone | ResNet-18 |
| Initialization | ImageNet V1 pretrained weights |
| Output classes | 2 |
| Loss | Cross-entropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 10 |
| Device selection | CUDA if available, otherwise CPU |
| Saved artifact | PyTorch model `state_dict` |

The training script evaluates the test loader after each epoch and prints loss and accuracy, then saves the final epoch's weights. It does not use a distinct validation set, select a best checkpoint, implement early stopping, save a structured metrics log, or set an explicit random seed for model initialization and data-loader shuffling. For this manuscript's exploratory evaluation, the locally available checkpoint was loaded strictly into the same ResNet-18 architecture on CPU and evaluated in inference mode on the local `dataset/test` folder, with `ImageFolder` ordering and batch size 32. The run processed 920 images in 21.559 seconds. The checkpoint's source run cannot be confirmed from repository records; therefore the folder cannot be claimed to be an untouched test cohort.

The application loads the checkpoint on CPU. If the file is unavailable, it constructs a ResNet-18 with ImageNet weights and a newly initialized two-output layer. The current loader also suppresses checkpoint-loading exceptions. These fallback states are not a trained tumor classifier and must not be used to report meaningful predictions.

### 2.5 Inference and output

The application performs a forward pass on one image and applies softmax to the two logits. It labels output index 1 as `TUMOR DETECTED` and index 0 as `HEALTHY SCAN`; the maximum softmax value is displayed as a percentage. This value is an uncalibrated model score, not a clinically validated probability or measure of accuracy.

The classifier is binary. A separate UI selector allows a user to choose Glioma, Meningioma, or Pituitary Tumor for predefined text/report content after a tumor prediction. This selector is not an output of the trained model.

### 2.6 Explainability and visual analytics

Grad-CAM [3] is computed from the last block of ResNet-18 (`model.layer4[-1]`). Gradients are spatially averaged to weight the activation maps; the weighted map is summed across channels, passed through ReLU, and normalized. The map is resized and overlaid on the input image for display.

Grad-CAM is an exploratory visualization. The repository does not compare maps with expert lesion segmentations, calculate localization metrics, or demonstrate that highlighted pixels correspond to tumor tissue.

The analytics page attaches forward hooks to selected ResNet layers and displays up to 16 feature maps. Its coordinate display finds the maximum heatmap location and applies a linear conversion to two values; the third value is randomly selected. There is no DICOM geometry, voxel spacing, orientation, anatomical registration, or MNI template. Therefore, the displayed coordinates and brain-region label are not validated MNI-space localization.

The optional 3D view samples bright pixels in the single input image and assigns random depth values. A heatmap mask can be overlaid as another point set. This visualization is illustrative only; it is not volumetric imaging, segmentation, or surgical planning.

### 2.7 User interface, reports, and generative-AI features

The patient-record page stores demonstration records in Streamlit session state. There is no persistent database or implemented authentication, authorization, audit system, or electronic health record integration.

The report module combines user/patient fields, the binary model output or user-selected label, a score, placeholder location/size values, and predefined text into a PDF. It does not generate a clinically validated radiology report. Medication and follow-up statements are static strings in source code rather than patient-specific recommendations.

When configured with `OPENROUTER_API_KEY`, the language-model module requests `google/gemini-2.0-flash-001` through OpenRouter. Image description sends the image and prompt to a remote service; chat sends text messages. The browser's Web Speech API can read generated text aloud. Responses are not validated and may be incorrect. No protected or identifiable image data should be transmitted without appropriate consent, authorization, and institutional/privacy review.

### 2.8 Exploratory local evaluation

The available `state_dict` was loaded with strict key matching into a ResNet-18 with a two-output fully connected layer. The nominal test set was loaded from `BrainTumorAI/dataset/test` using Torchvision `ImageFolder`, resized to 224 × 224 pixels, converted to tensors, and normalized using the ImageNet mean and standard deviation. Inference used batches of 32 on CPU with shuffling disabled. The predicted class was the argmax of the two logits. The tumor probability used for ROC-AUC was the softmax score for class index 1. We computed image-level accuracy, confusion matrix, tumor precision/recall/F1, no-tumor specificity, balanced accuracy, and ROC-AUC.

The accuracy interval is a two-sided 95% Wilson binomial interval computed from 521 correct classifications among 920 images. This interval does not account for possible dependence among images from the same patient. Patient identifiers, patient counts, dataset provenance, duplicate status, and checkpoint training history were unavailable; no patient-level or inferential performance claim is made.

The evaluation environment was Python 3.13.1, PyTorch 2.7.1+cu118, Torchvision 0.22.1+cu118, and scikit-learn 1.6.1 on Windows with an AMD Ryzen 7 7435HS CPU. CUDA was available in the environment, but this evaluation explicitly ran on CPU. The local checkpoint's SHA-256 was `779D87161A8118DF39A9518B8B02FC663ED95990CE1DC5450C0266FDCA3F3A97`. The dataset and checkpoint are ignored by Git and are not distributed with the repository.

## 3. Results

### 3.1 Software implementation

The implemented prototype provides an image-upload workflow, binary class-score display, Grad-CAM overlay, selected-layer activation maps, a synthetic three-dimensional visualization, static informational/report output, and optional remote generative-AI and browser speech features.

### 3.2 Classifier performance

The following are descriptive metrics from one run on the local test folder. They are not patient-level estimates and do not establish clinical performance.

| Metric | No-tumor class | Tumor class | Binary overall |
|---|---:|---:|---:|
| Test images | 399 | 521 | 920 |
| Test patients | Not available | Not available | Not available |
| Recall/sensitivity by class | 0.00% | 100.00% | — |
| Precision by class | 0.00% (undefined; zero-division convention) | 56.63% | — |
| F1 score by class | 0.00% | 72.31% | — |
| Specificity (tumor treated as positive) | — | — | 0.00% |
| Accuracy | — | — | 56.63% (Wilson 95% CI 53.41–59.80%) |
| Balanced accuracy | — | — | 50.00% |
| ROC-AUC | — | — | 0.487 |

For class-index order `[no_tumor, tumor]`, the confusion matrix (rows are true labels; columns are predicted labels) was:

|  | Predicted no tumor | Predicted tumor |
|---|---:|---:|
| True no tumor | 0 | 399 |
| True tumor | 0 | 521 |

The classifier returned tumor for all 920 images. Its accuracy therefore equals the tumor-class prevalence in this folder; the binary specificity of 0%, balanced accuracy of 50%, and ROC-AUC below 0.5 indicate no useful discrimination in this evaluation. The result is a warning about the local checkpoint, not a claim that every training run or future checkpoint will behave identically. No calibration analysis, threshold selection, external validation, or patient-level analysis was performed.

### 3.3 Explainability and usability

No expert review of Grad-CAM maps, quantitative localization analysis, accessibility evaluation, or usability study was conducted. The interface demonstrates those features in software only.

## 4. Discussion

### 4.1 Principal findings

The software implements an interactive binary image-classification workflow with exploratory visualizations. However, the one-run evaluation of the available local checkpoint produced an all-tumor prediction pattern: it detected all 521 tumor-folder images but also classified all 399 no-tumor-folder images as tumor. The resulting 56.63% image-level accuracy reflects class prevalence rather than useful discrimination. This finding argues against interpreting the checkpoint output as a functional diagnostic classifier.

### 4.2 Interpretation

The prototype illustrates one way to present a binary classifier and exploratory visualizations through a browser-based application. The current implementation does not establish that predictions generalize to independent patients or clinical settings. A visually plausible heatmap does not establish lesion localization, and an uncalibrated softmax value should not be interpreted as diagnostic certainty.

### 4.3 Limitations

Limitations apparent from the current repository include:

- incomplete dataset provenance and no tracked data citation;
- image-level, non-stratified splitting without patient grouping or duplicate detection;
- no separate validation set, best-checkpoint selection, or recorded final test evaluation;
- no explicit reproducibility seed for all random operations;
- a binary classifier despite subtype terminology in the user interface;
- fallback model loading that can silently leave a randomly initialized output head;
- no probability calibration or external validation;
- Grad-CAM without annotated localization validation;
- heuristic, partly random coordinates rather than MNI registration;
- a synthetic 3D visualization rather than volume reconstruction;
- static treatment text and placeholder report fields;
- session-only demonstration patient records rather than a secure clinical data system;
- optional remote processing of image/text without a validated clinical or privacy workflow.

In addition, the locally evaluated checkpoint's training run is not recorded, and the all-positive output pattern indicates that this specific local model artifact is unsuitable for meaningful classification in the evaluated folder. The local counts and measured outputs are not independently reproducible from a fresh clone because the dataset and checkpoint are not tracked.

### 4.4 Future work

Potential future work, contingent on research approvals and a defined study design, includes dataset provenance documentation; patient-level cohort splitting; external validation; calibrated predictions; clinically appropriate comparators; expert-reviewed localization evaluation; support for volumetric data with correct spatial metadata; persistence/security engineering; accessibility/usability evaluation; and prospective assessment. These are proposals, not existing capabilities.

## 5. Conclusion

NeuroVision AI is a software prototype that integrates binary brain-image classification with exploratory visual analytics and a Streamlit interface. In a single exploratory run, the available local checkpoint classified every image in the nominal test folder as tumor, yielding 50% balanced accuracy and no specificity for the no-tumor class. The artifact should not be used for diagnosis or treated as a useful detector. Because dataset provenance, patient independence, and checkpoint training history are unknown, these image-level results are descriptive and cannot establish generalizable diagnostic performance. Future evaluation requires a traceable dataset, a prespecified patient-independent design, a verified checkpoint, and appropriate clinical and ethical review.

## Declarations

- **Ethics approval:** Not reported in the repository. Authors must determine and state the applicable review/approval status before submission.
- **Consent:** Not reported in the repository; the applicable consent or waiver must be confirmed by the data provider and authors.
- **Data availability:** The local data folder is not tracked in Git. Its original source, license, and redistribution conditions were not established during this software review.
- **Code availability:** The project source is hosted at `https://github.com/stephenrodrick17-cloud/Neurovision-AI`. The exact commit used for the local evaluation should be recorded before submission.
- **Funding:** Funding information was not available in the inspected project materials; authors must confirm.
- **Conflicts of interest:** Conflict-of-interest declarations were not available in the inspected project materials; authors must confirm.
- **Author contributions:** Contributions were not established from the code review and must be completed by the authors.
- **Acknowledgments:** None are asserted in this draft.

## References

1. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016:770–778. doi:10.1109/CVPR.2016.90.
2. Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L. ImageNet: A large-scale hierarchical image database. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2009:248–255. doi:10.1109/CVPR.2009.5206848.
3. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2017:618–626. doi:10.1109/ICCV.2017.74.

The source dataset's primary citation could not be identified from the repository and must be added once its provenance is established. Verify citation metadata and adapt formatting to the target journal before submission.

## Before submission checklist

- [ ] Confirm dataset source, citation, license, cohort definition, and label provenance.
- [ ] Rebuild splits so images from one patient cannot appear in multiple partitions.
- [ ] Check duplicates and record all exclusions and preprocessing failures.
- [ ] Separate train, validation, and final test data; do not tune on the final test cohort.
- [ ] Record code commit, dependency versions, hardware, seeds, checkpoint hash, and full training logs.
- [ ] Verify the checkpoint used for reported results is the checkpoint produced by the documented run.
- [ ] Re-evaluate a verified checkpoint and calculate patient-level metrics and confidence intervals from saved predictions.
- [ ] Add calibration and external validation if making probability or generalization claims.
- [ ] Remove or explicitly qualify unvalidated MNI, 3D, treatment, subtype, and clinical claims.
- [ ] Complete ethics, privacy, consent, funding, authorship, and data-availability statements.
- [ ] Have a domain expert and statistician review the manuscript and claims.
