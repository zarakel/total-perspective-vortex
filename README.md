# Total Perspective Vortex - BCI Motor Imagery Classification

## 1. Project Overview and Summary

The **Total Perspective Vortex** project is a Brain-Computer Interface (BCI) application designed to classify electroencephalographic (EEG) signals from motor imagery (for instance, imagining left hand movement versus right hand movement).

By leveraging the reference dataset **PhysioNet EEGBCI** (109 subjects, 64 EEG channels) as well as the **BCI Competition IV-2a** dataset, the system implements a complete processing pipeline: from temporal and spatial filtering of raw signals to classification using machine learning. The main goal is to design a robust scikit-learn pipeline achieving a classification accuracy of >= 60% (the optimized pipeline achieves an average of ~74% on PhysioNet and ~81% on BCI Competition IV-2a).

The project stands out by integrating key algorithms rewritten entirely from scratch (LDA classification and generalized eigenvalue decomposition), demonstrating mastery of the underlying mathematical foundations of BCIs.

---

## 2. Techniques and Acquired Skills

This project covers a wide range of scientific, mathematical, and software engineering skills:

### A. EEG Digital Signal Processing
*   **Physiological Temporal Filtering**: Design and application of FIR (Finite Impulse Response) bandpass filters via [src/preprocessing.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/preprocessing.py) in the key spectral bands of motor imagery: the **mu band (8-12 Hz)** and the **beta band (18-30 Hz)**, which are the main sites for event-related desynchronization/synchronization (ERD/ERS) phenomena.
*   **Electrical Noise Removal**: Implementation of a notch filter at 50 Hz (and its harmonics at 100 Hz) to eliminate industrial electrical power line interference.
*   **Spatial Reference Reduction (CAR)**: Application of the Common Average Reference to subtract the instantaneous average potential of all sensors to mitigate global noise.
*   **Time-Frequency Analysis**: Estimation of Power Spectral Density (PSD) using Welch's method and extraction of energy features using Discrete Wavelet Transform (with Daubechies `db4` wavelets).

### B. Machine Learning and Modeling (BCI)
*   **Supervised Spatial Filtering (CSP)**: Design of a custom [CSP (Common Spatial Patterns)](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/csp_custom.py) transformer adhering to the scikit-learn interfaces (`BaseEstimator`, `TransformerMixin`). CSP projects multichannel signals to maximize the variance of one class while minimizing that of the other.
*   **Filter Bank Common Spatial Patterns (FBCSP)**: Implementation of an FBCSP pipeline as a `FeatureUnion` running multiple independent CSPs in parallel over different frequency sub-bands (mu, low-beta, high-beta).
*   **Covariance Matrix Regularization**: Application of shrinkage techniques (linear regularization) to improve the reliability of covariance matrix estimation on datasets with few epochs.
*   **Hyperparameter Search and Cross-Validation**: Use of `StratifiedKFold` and `GridSearchCV` to optimize the number of CSP components, the regularization parameter, and the choice of classifier.
*   **BCI Illiteracy Analysis**: In-depth statistical analysis of the impact of non-responder subjects (accuracy < 60%) on the overall and per-subject average.

### C. Algorithms and Applied Mathematics (Bonus "From Scratch")
*   **Custom Eigen-Decomposition**: Implementation in [src/eigen_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/eigen_custom.py) of an eigenvalue/eigenvector decomposition algorithm for symmetric matrices:
    *   **Householder Tridiagonalization** to transform the dense symmetric matrix into a tridiagonal matrix.
    *   **QR Iterations with Wilkinson shifts** and Givens rotations to achieve cubic convergence towards the eigenvalues.
    *   **Cholesky Factorization** to transform the generalized problem $Av = \lambda Bv$ into a standard symmetric problem.
*   **Custom LDA Classifier**: Development in [src/lda_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/lda_custom.py) of a Linear Discriminant Analysis classifier (Fisher's LDA) solved analytically (computation of class means, within-class scatter matrix, covariance regularization by shrinkage, and probabilistic predictions via a sigmoid function).

### D. Software Engineering and Real-Time Systems
*   **Real-Time Flow Simulation**: Implementation of an EEG epoch streaming generator in [src/stream_simulator.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/stream_simulator.py), measuring and ensuring prediction latency is below the critical threshold of 2 seconds.
*   **Containerization and GUI Support**: Configuration of a complete Docker environment ([dockerfile](file:///home/jbuan/sgoinfre/total-perspective-vortex/dockerfile) and [docker-compose.yml](file:///home/jbuan/sgoinfre/total-perspective-vortex/docker-compose.yml)) with X11 server forwarding to enable interactive visualization of EEG signals and confusion matrices from within the container.

---

## 3. Project Architecture

```
Raw EEG (64 channels x N time points)
    │
    ▼
FIR Bandpass Filtering (8-30 Hz)        <- src/preprocessing.py
    │
    ▼
Epoching                                <- src/loader.py
    │
    ▼
┌─── sklearn Pipeline ──────────────────────────────────┐
│  Custom CSP (BaseEstimator + TransformerMixin)        │  <- src/csp_custom.py
│    -> Resolves the generalized eigenvalue problem     │
│    -> Projects onto spatial filters                   │
│    -> Returns log-variances                           │
│  StandardScaler                                       │
│  Classifier (LogisticRegression or custom LDA)        │  <- src/lda_custom.py (bonus)
└───────────────────────────────────────────────────────┘
    │
    ▼
Prediction: "left hand" or "right hand"
```

### Alternative Pipeline (FeatureExtractor)
Can be enabled via `--use-features`: replaces CSP with PSD + wavelet feature extraction.
```
Epochs -> FeatureExtractor (band PSD + wavelets) -> StandardScaler -> Classifier
```

---

## 4. Source File Structure

| File | Role |
|---|---|
| [tpv.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/tpv.py) | Main entry point - CLI (`train`, `predict`, `evaluate-all`). |
| [src/loader.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/loader.py) | Automatic download of PhysioNet data via MNE, normalization, and epoching of signals. |
| [src/preprocessing.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/preprocessing.py) | Bandpass filtering, power line notch filtering, and graphical visualizations (raw/PSD). |
| [src/csp_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/csp_custom.py) | Implementation of the CSP transformer from scratch (scikit-learn compatible). |
| [src/pipeline_model.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/pipeline_model.py) | Assembly of the sklearn pipeline, FBCSP support, and GridSearch. |
| [src/features.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/features.py) | Alternative feature extractor (PSD via Welch, discrete wavelets, and CAR). |
| [src/stream_simulator.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/stream_simulator.py) | Epoch streaming generator to validate real-time latency constraints. |
| [src/lda_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/lda_custom.py) | **Bonus**: Linear Discriminant Analysis classifier coded from scratch. |
| [src/eigen_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/eigen_custom.py) | **Bonus**: Eigen-decomposition algorithms (Householder, QR, Givens, Cholesky) coded from scratch. |
| [dockerfile](file:///home/jbuan/sgoinfre/total-perspective-vortex/dockerfile) | Docker image (Python 3.11) containing all required scientific computing libraries. |
| [docker-compose.yml](file:///home/jbuan/sgoinfre/total-perspective-vortex/docker-compose.yml) | Docker service with X11 forwarding for graphical display. |

---

## 5. Main Commands

### Building the Docker Image
```bash
docker compose build
```

### Training a Model (Train)
```bash
# Classic training (CSP + LogisticRegression) on a single subject
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib

# Training with graphical visualizations (raw, filtered, and PSD signals)
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib --show-raw

# Training with rewritten components (custom LDA + custom eigen)
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib \
  --use-custom-clf --use-custom-eigen

# Training with hyperparameter tuning via GridSearch
docker compose run --rm matplotlib python tpv.py train \
  --subject 1 --runs 4 8 12 --model-path model.joblib --tune
```

### Simulated Stream Prediction (Predict)
*Test runs must be distinct from training runs.*
```bash
docker compose run --rm matplotlib python tpv.py predict \
  --subject 1 --runs 3 7 11 --model-path model.joblib
```

### Global Evaluation (Evaluate-all)
Allows testing the FBCSP + LDA pipeline across all subjects and experiments.
```bash
# Full evaluation on all 109 subjects of PhysioNet
docker compose run --rm matplotlib python tpv.py evaluate-all

# Fast test restricted to the first 5 subjects
docker compose run --rm matplotlib python tpv.py evaluate-all --max-subjects 5

# Evaluation on the BCI Competition IV-2a dataset (9 subjects, 22 channels)
docker compose run --rm matplotlib python tpv.py evaluate-all --dataset bci4-2a
```

---

## 6. Technical Details by Module

### [src/preprocessing.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/preprocessing.py) - Filtering & Visualization
*   **Bandpass Filtering (8-30 Hz)**: Retains only the physiological frequency bands of interest for Motor Imagery (mu and beta). Implemented using MNE's FIR filter (`firwin`).
*   **Notch Filter (50/100 Hz)**: Eliminates power line noise and its first harmonic.
*   **Visualization**: Tools to display the raw/filtered temporal signal and the Power Spectral Density (PSD) spectrum.

### [src/loader.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/loader.py) - Loading & Epoching
*   **Standardization**: Automatic download and application of a standard electrode montage (`standard_1005`).
*   **Channel Selection**: Automatic filtering to restrict processing to the **21 sensorimotor channels** by default (FC5/3/1/z/2/4/6, C5/3/1/z/2/4/6, CP5/3/1/z/2/4/6), optimizing the signal-to-noise ratio for motor imagery.
*   **Data Augmentation**: Option for data augmentation via overlapping temporal windowing (beware of data leakage risks if applied before train/test splitting).

### [src/csp_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/csp_custom.py) - Common Spatial Patterns
*   **Covariance Calculation**: Normalized estimation by the trace to balance the importance of each epoch.
*   **Regularization (Shrinkage)**: Linear combination of the estimated covariance matrix with a uniform diagonal matrix: $\Sigma_{\text{reg}} = (1-\lambda)\Sigma + \lambda \frac{\text{Tr}(\Sigma)}{d} I$.
*   **Feature Extraction**: Spatial projection and extraction of log-variances from the spatially filtered signals.

### [src/features.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/features.py) - Alternative Extraction
*   **Frequency Band PSD**: Welch's computation on standard brainwave bands.
*   **Discrete Wavelets**: Cascaded decomposition (Daubechies 4 wavelet with 3 detail levels) to extract energy and dispersion at different time and frequency resolutions.

---

## 7. How the "From Scratch" Algorithms Work (Bonus)

### Generalized Eigenvalue Decomposition ([src/eigen_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/eigen_custom.py))
To solve the spatial problem $A v = \lambda B v$ where $A$ and $B$ are real symmetric covariance matrices:
1.  **Cholesky Factorization**: B is decomposed into $B = L L^T$.
2.  **Standard Transformation**: We define $C = L^{-1} A L^{-T}$, reducing the problem to a classical symmetric eigenvalue problem $C u = \lambda u$ (where $v = L^{-T} u$).
3.  **Householder Tridiagonalization**: Using successive Householder reflections, off-diagonal elements are zeroed out to transform $C$ into a symmetric tridiagonal matrix $T = Q_0^T C Q_0$.
4.  **QR Algorithm with Wilkinson Shifts**: Successive QR factorizations are performed on $T$. Using Wilkinson shifts significantly accelerates convergence to the diagonal form (eigenvalues on the diagonal).
5.  **Givens Rotations**: Implemented to perform QR factorization steps on the tridiagonal matrix efficiently (linear complexity per iteration).

### Linear Discriminant Analysis ([src/lda_custom.py](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/lda_custom.py))
1.  **Class Means**: Computation of feature centroids $\mu_1$ and $\mu_2$ for both classes.
2.  **Within-Class Scatter Matrix ($S_w$)**: Weighted sum of the covariance matrices of each class.
3.  **Ledoit-Wolf Regularization**: Adjustment of $S_w$ to guarantee its non-singularity.
4.  **Projection Vector ($w$)**: Resolving the linear system $S_{w} w = \mu_1 - \mu_2$.
5.  **Classification**: Projection of new data points and comparison against the median threshold: $x \mapsto w^T x - \text{threshold}$.

---

## 8. Global Evaluation Mode (`evaluate-all`)

The script evaluates each subject across **4 distinct experimental tasks** using a 5-fold cross-validation (`StratifiedKFold`):

| Experiment | Associated Task | PhysioNet Runs |
|---|---|---|
| **0** | Motor Execution (Actual movement of left/right fist) | 3, 7, 11 |
| **1** | Motor Imagery (Imagining movement of left/right fist) | 4, 8, 12 |
| **2** | Motor Execution (Actual movement of fists/feet) | 5, 9, 13 |
| **3** | Motor Imagery (Imagining movement of fists/feet) | 6, 10, 14 |

### Optimized Classification Pipeline
To achieve maximum and stable performance, the default pipeline configured in `evaluate-all` uses:
*   A wide **4-40 Hz** input bandpass filter.
*   **FBCSP** multi-band extraction (bands: 8-12 Hz, 12-20 Hz, and 20-30 Hz).
*   Adaptive artifact rejection removing signals with anomalous amplitudes ($> 200\,\mu\text{V}$).
*   Classification via **LDA with automatic shrinkage**.

---

## 9. Required Dependencies

The project's dependencies are listed in [src/requirements.txt](file:///home/jbuan/sgoinfre/total-perspective-vortex/src/requirements.txt) and automatically installed in the Docker container:
*   `numpy` & `scipy` (Matrix and scientific computing)
*   `mne` & `moabb` (EEG data loading and manipulation)
*   `scikit-learn` (ML pipelines, scalers, metrics)
*   `matplotlib` (Graphical display)
*   `joblib` (Model saving)
*   `PyWavelets` (Wavelet decomposition)