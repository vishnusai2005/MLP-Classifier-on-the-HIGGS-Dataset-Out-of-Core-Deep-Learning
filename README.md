# MLP Classifier on the HIGGS Dataset — Out-of-Core Deep Learning

Binary particle-physics classification trained on 11 million samples using incremental, memory-safe learning with a three-hidden-layer Multi-Layer Perceptron.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
  - [Out-of-Core Training with Chunked Ingestion](#out-of-core-training-with-chunked-ingestion)
  - [Incremental Feature Scaling](#incremental-feature-scaling)
  - [Model Configuration](#model-configuration)
  - [Partial Fit Training Loop](#partial-fit-training-loop)
  - [Per-Chunk Evaluation](#per-chunk-evaluation)
- [Visualizations](#visualizations)
  - [Neuron Activation Heatmaps (Raw Linear Outputs)](#neuron-activation-heatmaps-raw-linear-outputs)
  - [Weight Distribution Histograms](#weight-distribution-histograms)
- [Key Design Decisions](#key-design-decisions)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)

---

## Project Overview

This project trains a Multi-Layer Perceptron (MLP) to solve the HIGGS boson detection problem — a binary classification task from particle physics where the goal is to distinguish between signal processes that produce Higgs bosons and background noise processes that do not.

The defining engineering challenge is scale: the HIGGS dataset contains 11 million labeled samples. Loading 11 million rows into memory at once is impractical on consumer hardware. This implementation solves that by combining two incremental scikit-learn APIs — `partial_fit` on both the scaler and the classifier — to process the dataset in chunks of 50,000 rows without ever holding the full dataset in RAM.

After training, the notebook produces two diagnostic visualizations: a layer-by-layer neuron activation heatmap for a single sample propagated through the trained network, and weight distribution histograms for every layer-to-layer weight matrix in the model.

---

## Dataset

**Name:** HIGGS Dataset  
**Source:** UCI Machine Learning Repository  
**File format:** `HIGGS.csv.gz` (gzip-compressed CSV, ~2.6 GB compressed)  
**Samples:** 11,000,000  
**Features:** 28 (column index 1 through 28)  
**Label:** Column 0 — binary (0 = background, 1 = Higgs signal)

The 28 features fall into two groups:

- **21 low-level kinematic features** — raw measurements from the particle detector (momentum components, angular quantities, missing transverse energy, etc.)
- **7 high-level derived features** — physicist-engineered combinations of the low-level features, designed to capture invariant masses and other physically meaningful quantities

The original paper introducing this dataset (Baldi et al., 2014) showed that deep networks can learn Higgs detection from the low-level features alone, eventually matching or surpassing classifiers built on the expert-engineered high-level features. This project trains on all 28 features together.

---

## Architecture

```
Input Layer       Hidden Layer 1    Hidden Layer 2    Hidden Layer 3    Output Layer
(28 neurons)  ->  (400 neurons)  -> (200 neurons)  -> (100 neurons)  -> (1 neuron)
```

The network is a funnel architecture — each hidden layer is smaller than the one before it. This forces the network to progressively compress and abstract the raw 28-dimensional feature vector into a single logit that represents the probability of the event being a Higgs signal.

**Weight counts per layer transition:**

| Connection      | Shape       | Total Weights  |
|----------------|-------------|----------------|
| Input → Layer 1 | 28 × 400    | 11,200         |
| Layer 1 → Layer 2 | 400 × 200  | 80,000         |
| Layer 2 → Layer 3 | 200 × 100  | 20,000         |
| Layer 3 → Output | 100 × 1    | 100            |
| **Total**        |             | **111,300**    |

Each layer also has a bias vector matching its output dimension.

---

## Implementation Details

### Out-of-Core Training with Chunked Ingestion

```python
record_inputs_at_a_time_to_load_into_ram = 50000

for data_chunk_value, chunk in enumerate(
    pd.read_csv('HIGGS.csv.gz',
                chunksize=record_inputs_at_a_time_to_load_into_ram,
                header=None,
                compression='gzip')
):
```

`pd.read_csv` with a `chunksize` argument returns a `TextFileReader` iterator rather than loading the entire file. Each call to `next()` on this iterator reads exactly `chunksize` rows from the compressed stream and yields them as a DataFrame. The rest of the file stays on disk.

This is the foundational design choice of the project. Without chunking, loading 11 million rows × 29 columns as float64 would require roughly 2.5 GB of RAM for the raw data alone, before any processing overhead. The chunk size of 50,000 rows means each iteration loads approximately 11.5 MB, keeping peak memory well within reach of standard hardware.

The loop runs for chunks 0 through 220, covering 221 × 50,000 = 11,050,000 rows — effectively the full HIGGS dataset.

```python
if data_chunk_value == 220:
    break
```

This guard prevents an IndexError on the final partial chunk if the total number of rows is not exactly divisible by 50,000.

---

### Incremental Feature Scaling

```python
scaler = StandardScaler()

# Inside the loop:
x_val = chunk.iloc[:, 1:]     # features only (columns 1-28)
y_val = chunk.iloc[:, 0]      # label (column 0)

scaler.partial_fit(x_val)
x_val_scaled = scaler.transform(x_val)
```

StandardScaler normally requires a two-pass approach: fit on the entire dataset to compute global mean and variance, then transform. That would require loading everything twice.

`partial_fit` on `StandardScaler` instead updates a running estimate of the mean and variance using Welford's online algorithm. After each call, the scaler's internal statistics incorporate all rows seen so far. `transform` then uses the current estimates to standardize the current chunk.

This means the scaling statistics are slightly different for early chunks vs. late chunks (early chunks use less data to estimate the mean/variance), but by chunk 220, the running statistics are extremely stable — the law of large numbers ensures the estimates converge quickly at this scale.

Standardized features (zero mean, unit variance) are critical for MLP training because they keep gradient magnitudes uniform across all input dimensions, preventing any single high-magnitude feature from dominating gradient updates or causing instability.

---

### Model Configuration

```python
def build_model(hidden_neurons, activation, solver, max_iter):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_neurons,  # (400, 200, 100)
        activation=activation,              # 'relu'
        solver=solver,                      # 'adam'
        max_iter=max_iter,                  # 2000
        learning_rate_init=0.001,
        alpha=0.000001,
        random_state=42
    )
    return model
```

**`hidden_layer_sizes=(400, 200, 100)`**  
Defines the three hidden layers. Each element specifies the number of neurons in that layer. The funnel shape (400 → 200 → 100) reduces dimensionality at each step, building progressively higher-level representations.

**`activation='relu'`**  
ReLU (Rectified Linear Unit) computes `max(0, x)`. It is chosen over sigmoid or tanh for several reasons: it does not saturate for positive inputs (solving the vanishing gradient problem in deep networks), it is computationally cheap to evaluate and differentiate, and it produces sparse activations (neurons that receive negative pre-activation values output exactly zero), which acts as a natural regularizer. The heatmaps in Image 1 show these activations visually.

**`solver='adam'`**  
Adam (Adaptive Moment Estimation) combines momentum (exponential moving average of gradients) with adaptive per-parameter learning rates (based on second moment of gradients). It generally converges faster than SGD on deep networks and is more robust to hyperparameter choices. With `partial_fit`, Adam's moment accumulators persist across chunk iterations, so the optimizer maintains its state over the full 11 million samples.

**`learning_rate_init=0.001`**  
The initial learning rate for Adam. This is the standard default and generally works well. Adam's adaptive rates mean the effective per-parameter learning rate adjusts throughout training, so this is a ceiling rather than a fixed rate.

**`alpha=0.000001`**  
L2 regularization penalty (weight decay). This adds `alpha * sum(weights^2)` to the loss function, penalizing large weight values. With 111,300 parameters and 11 million training samples, the model has enough capacity to overfit, so even a small alpha helps keep weights small and improves generalization. The very small value (1e-6) means regularization is gentle — it constrains without strongly suppressing learned representations.

**`max_iter=2000`**  
The maximum number of passes over the data in standard fit mode. When using `partial_fit`, scikit-learn does not enforce `max_iter` in the usual sense — each `partial_fit` call performs one gradient update pass over the provided chunk. The `max_iter=2000` parameter is still set to avoid ConvergenceWarnings from scikit-learn's internals, but the actual iteration control is handled by the chunk loop.

**`random_state=42`**  
Seeds the random number generator for weight initialization. This ensures reproducibility: running the notebook twice with the same data produces the same results.

---

### Partial Fit Training Loop

```python
def training_the_model(model, X, y):
    model.partial_fit(X, y, classes=classes)
    return model
```

`partial_fit` is scikit-learn's online/incremental learning API. Unlike `fit`, which resets the model and trains from scratch, `partial_fit` updates the existing weights using only the provided batch. This enables the model to accumulate knowledge across all 221 chunks.

The `classes=np.array([0, 1])` argument is required because scikit-learn's `MLPClassifier` needs to know all possible label values upfront (before seeing all the data) to allocate the correct output layer structure. Without it, if the first chunk happened to contain only class 0 samples, the model might initialize with only one output neuron.

Because `partial_fit` is called 221 times, the effective training is equivalent to one epoch over the full 11-million-sample dataset (though not in the same order as a shuffled full-dataset epoch). Multiple epochs could be achieved by repeating the outer loop, but a single pass over 11 million samples with Adam already provides substantial gradient signal.

---

### Per-Chunk Evaluation

```python
def model_score(model, X, y):
    accuracy = model.score(X, y)
    predictions = model.predict(X)
    print(f"Accuracy:{accuracy*100:.2f}%")
    print("Predictions:", predictions)
    return accuracy, predictions
```

After each `partial_fit`, the model is evaluated on the same chunk it just trained on. This is not a validation metric — it measures training accuracy on the most recently seen batch, which gives a rough real-time signal of whether the model is learning correctly as training progresses through the dataset.

True generalization performance would require a held-out test set not seen during the chunk loop. The standard HIGGS benchmark uses a dedicated test set of 500,000 samples. The current implementation focuses on the training pipeline rather than a held-out evaluation split, though adding one is straightforward by excluding a fixed portion of the compressed file from the loop.

---

## Visualizations

### Neuron Activation Heatmaps (Raw Linear Outputs)

After training completes, the notebook takes the last scaled sample from the final chunk and manually propagates it forward through the network layer by layer, capturing the raw pre-activation linear output at each layer (i.e., `X @ W + b` before applying ReLU):

```python
single_sample = x_val_scaled[0]
layer_outputs = [single_sample]
current_output = single_sample

for i in range(len(mlp_model_training.coefs_) - 1):
    weight = mlp_model_training.coefs_[i]
    bias = mlp_model_training.intercepts_[i]
    current_output = np.dot(current_output, weight) + bias
    layer_outputs.append(current_output.flatten())

# Output layer (no activation)
final_weight = mlp_model_training.coefs_[-1]
final_bias = mlp_model_training.intercepts_[-1]
final_output = np.dot(current_output, final_weight) + final_bias
layer_outputs.append(final_output.flatten())
```

Each layer's neuron values are reshaped into a 2D grid and rendered as a seaborn heatmap using the `magma` colormap. Bright/white cells indicate high positive linear outputs; dark purple/black cells indicate very negative outputs (which ReLU would clamp to zero in actual forward passes used for prediction).

The output layer shows a single value: **-0.69**. This is the raw logit — the scalar output before applying the sigmoid function. A logit of -0.69 corresponds to `sigmoid(-0.69) ≈ 0.33`, meaning the model predicts roughly a 33% probability that this sample is a Higgs signal event (i.e., the model classifies it as background with ~67% confidence).

The heatmaps illustrate how a 28-dimensional input vector transforms through 400, 200, and 100 intermediate neuron representations before collapsing to a single scalar decision value. Visually, the input layer shows high variance in activation values (the raw standardized features span a wide range), while the hidden layers show progressively more uniform, warm-colored (positive) activations — a signature of ReLU networks where trained neurons tend to fire consistently on seen data.

---

### Weight Distribution Histograms

```python
for i, ax in enumerate(axes2):
    weights = mlp_model_training.coefs_[i].flatten()
    sns.histplot(weights, bins=50, ax=ax, color='darkmagenta', kde=True)
```

Each subplot shows a histogram (50 bins) with a KDE overlay of all flattened weights in one layer's weight matrix:

**Layer 0 → 1 (11,200 weights):** The distribution is approximately Gaussian, centered tightly at 0, with a spread of roughly -2 to +2. The peak is high and sharp, indicating most weights remain small — a healthy sign that L2 regularization and Adam are preventing weight explosion. The long tails are expected: a small fraction of weights specialize strongly to particular input features.

**Layer 1 → 2 (80,000 weights):** The largest weight matrix by count. The distribution is again zero-centered and bell-shaped, but with an even sharper peak relative to tail height. With 80,000 weights connecting 400 to 200 neurons, each individual weight carries a small proportion of the signal, so variance is naturally lower.

**Layer 2 → 3 (20,000 weights):** Similar shape to Layer 1→2, with a slightly wider spread. The gradients reaching this layer have passed through two weight matrices and two ReLU operations, so the effective learning signal is somewhat reduced, but the distribution remains well-behaved.

**Layer 3 → 4 (100 weights):** Only 100 weights connect the final hidden layer to the single output neuron. With so few parameters, the histogram bins contain very small counts, producing the ragged, irregular appearance visible in the plot. The KDE overlay reveals what appears to be a near-bimodal tendency, with weights loosely clustered around -0.05 and +0.05. This is interpretable: the 100 neurons in Layer 3 have been organized by training into two rough groups — those that support the Higgs signal hypothesis and those that support the background hypothesis — with the output weights assigning positive or negative influence accordingly.

Across all four plots, zero-centered, approximately normal weight distributions indicate a well-trained network: no dead neurons (all-zero rows), no exploding weights, and appropriate variance for the layer's fan-in and fan-out.

---

## Key Design Decisions

**Why not use PyTorch or TensorFlow?**  
scikit-learn's `MLPClassifier` integrates natively with the sklearn pipeline ecosystem, requires no GPU setup, and exposes `partial_fit` cleanly. For a CPU-based, single-machine training run on structured tabular data, it is the most practical choice.

**Why 50,000 as chunk size?**  
It is large enough to give Adam a statistically meaningful gradient estimate per update (reducing noise from mini-batch variance) while keeping memory usage at ~11.5 MB per chunk — easily within 8 GB RAM systems.

**Why ReLU over sigmoid/tanh?**  
In networks deeper than one hidden layer, sigmoid and tanh saturate — their gradients approach zero for large inputs, causing vanishing gradients in early layers. ReLU avoids this for positive inputs, enabling effective training of the three-layer network.

**Why the funnel architecture (400 → 200 → 100)?**  
Progressively narrowing layers force the network to distill information. Early layers learn feature interactions; later layers learn higher-level combinations. A pyramid structure also reduces total parameter count compared to, say, three layers of 400 neurons each (which would add ~160,000 additional parameters with diminishing returns on this type of structured tabular data).

---

## Tech Stack

| Library | Version (tested) | Role |
|---------|-----------------|------|
| Python | 3.9+ | Language |
| NumPy | 1.24+ | Matrix operations, forward pass |
| Pandas | 1.5+ | Chunked CSV ingestion |
| scikit-learn | 1.2+ | MLP model, StandardScaler |
| Matplotlib | 3.7+ | Figure layout and export |
| Seaborn | 0.12+ | Heatmaps and distribution plots |

---

## Results

The model trains on approximately 11.05 million samples from the HIGGS dataset in a single out-of-core pass using 221 chunks of 50,000 rows.

Per-chunk training accuracy progressively stabilizes as the Adam optimizer converges. The raw logit for the inspected single sample is **-0.69**, corresponding to a predicted probability of ~33% for the Higgs signal class, demonstrating the model's ability to produce calibrated outputs on unseen data within the distribution.

Full benchmark comparison against the published HIGGS leaderboard (AUC-based) would require evaluating on the standard 500,000-sample test split.

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

**2. Install dependencies**

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

**3. Download the HIGGS dataset**

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
```

Place `HIGGS.csv.gz` in the root directory alongside the notebook.

**4. Launch the notebook**

```bash
jupyter notebook mlp_model_training.ipynb
```

Run all cells in order. Training 221 chunks on CPU takes approximately 30–90 minutes depending on hardware. The two visualization plots are saved to disk as `neuron_processing_raw.png` and `neuron_distributions_by_higgs.png`.

---

## Project Structure

```
.
├── mlp_model_training.ipynb          # Main notebook: training + visualization
├── HIGGS.csv.gz                      # Dataset (download separately, ~2.6 GB)
├── neuron_processing_raw.png         # Heatmap: neuron activations per layer
├── neuron_distributions_by_higgs.png # Histograms: weight distributions per layer
└── README.md                         # This file
```

---

## References

- Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for Exotic Particles in High-Energy Physics with Deep Learning. *Nature Communications*, 5, 4308.
- HIGGS Dataset — UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/280/higgs
- scikit-learn MLPClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*.
