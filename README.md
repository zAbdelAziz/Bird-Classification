# Bird-Classification
Audio Classification of a highly skewed audio dataset, as a part of the pattern classification project lab at JKU

## Dataset
Collaborative audio labeling was done by approximately 300 students to identify 7 labels, by selecting different parts of 1200 twenty-second bird recordings, Each recording was cautiously labeled by 3–6 annotators, and recordings were split into fragments of 200 ms, resulting in 120,000 training examples. For each fragment, the label and precomputed audio features were then preprocessed.

#### Pre-Processing
- All features are computed for frames of 1024 samples (≈ 43 ms) overlapped to give 70 frames per second.
- Each spectral feature is computed twice, [Raw / Filtered] spectrograms.
- In total, for each sample (200 ms) 548 features were calculated.

#### Features
* Mel Spectrograms
* Timbre [MFCC] (Averaged over different scales)
* Spectral Flatness, Centroid, Flux and Contrast
*  Energy, Power, ZCR
*  Yin

## Data Exploration Tasks
Extensive data analysis techniques were applied to the dataset supplied including
* Annotation Agreement Analysis.
* Data Balance, Inter/Intra Class Variation.
* Feature characteristics: Distribution, Correlation, Redundancy.
* Feature/Label overlap and relationships to identify the relevant features for classification.

## Explored Models:
* General Linear Models
* K-Nearest Neighbors
* Linear / Quadratic Discriminant Analysis
* Random Forest
* Feed-Forward Neural Networks
* Convolutional Neural Networks [1D, 2D] (Constant and Funnel AutoEncoders)

## Evaluation Metrics:
* Negative Log-Likelihood [NLL]
* Balanced Accuracy
* Weighted mean Average Precision-Recall
* Weighted F1-Score

Please check the reports for more insight into the procedure and implementation of each task.
