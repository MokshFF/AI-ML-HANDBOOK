# Python for Machine Learning: Engineering & Scientific Computing

A rigorous guide to Python syntax, numerical vectorization with NumPy, tabular feature manipulation with Pandas, visualization mechanics with Matplotlib, and software patterns for machine learning pipelines.

---

## 1. Python Fundamentals for ML

### 1.1 Intuition
Python is the standard language of machine learning because it couples concise, expressive high-level syntax with high-performance C and Fortran backends (such as NumPy, PyTorch, and SciPy). In ML engineering, Python acts as the orchestrator: preparing data, expressing mathematical equations, dispatching compute kernels to CPUs and GPUs, and exposing serving APIs.

### 1.2 Formal Definition & Execution Model
Python is a dynamically typed, garbage-collected language executed via bytecode interpretation within the CPython runtime. The CPython Global Interpreter Lock (GIL) serializes thread execution within the Python interpreter; however, numerical libraries like NumPy and PyTorch release the GIL during heavy compute kernels (BLAS, LAPACK, cuBLAS).

### 1.3 Key Concepts & Small Examples

#### Functional Constructs & Comprehensions
List and dictionary comprehensions offer cleaner and often faster iteration than manual `list.append()` calls because bytecode loops run in optimized C routines:
```python
# Efficient sample batching indexing
batch_sizes = [32, 64, 128, 256]
dim_to_latency = {bs: bs * 0.012 for bs in batch_sizes}
```

#### Generators for Large Datasets
Generators produce values on demand using `yield`, preventing the exhaustion of RAM when streaming millions of records:
```python
def stream_lines(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()
```

### 1.4 Common Mistakes
- **Mutating Default Arguments**: Using `def fn(weights=[])` causes the same list instance to persist across calls. Always use `weights: list | None = None` and instantiate inside the body.
- **Copy vs. View Confusion**: Slicing native Python lists produces a *shallow copy*, whereas slicing a NumPy array produces a *view*. Mutating an array slice directly alters the source tensor.

---

## 2. NumPy: Numerical Foundations & Vectorization

### 2.1 Intuition
CPython objects carry substantial overhead (a 64-bit integer takes 28 bytes due to type pointers and reference counts). Native Python loops suffer from pointer dereferencing and dynamic type checking at every iteration. NumPy stores arrays as contiguous memory buffers (SIMD-friendly) with a fixed data type (`float32`, `int64`), allowing processors to vectorize calculations across multiple registers simultaneously.

### 2.2 Mathematical Representation & Memory Stride
A NumPy $N$-dimensional array is defined by:
1. **Memory Pointer**: Address of the raw data block.
2. **Shape Tuple**: $(d_0, d_1, \dots, d_{k-1})$ specifying dimensional bounds.
3. **Strides Tuple**: $(s_0, s_1, \dots, s_{k-1})$ specifying the byte offset required to advance one unit in dimension $i$.

The memory address of element $(i_0, i_1, \dots, i_{k-1})$ is computed as:
$$\text{Address} = \text{DataPointer} + \sum_{j=0}^{k-1} i_j \cdot s_j$$

### 2.3 Broadcasting Rules
Two shapes are compatible for element-wise operations if, aligning dimensions from right to left (trailing dimensions):
1. The dimensions are equal, OR
2. One of the dimensions is $1$.

```python
import numpy as np

# Shape (B, D) + Shape (1, D) -> Shape (B, D)
batch_features = np.ones((64, 128))  # 64 samples, 128 dimensions
bias_vector = np.zeros((1, 128))     # 1 bias per feature
output = batch_features + bias_vector  # Broadcasted along axis 0
```

### 2.4 Vectorization Implementation
Comparing manual scalar accumulation against vectorized matrix dot product:
```python
# Dot product implementation: O(N) vectorized in C BLAS
x = np.random.randn(100_000)
w = np.random.randn(100_000)

# Unvectorized (Avoid)
dot_loop = sum(xi * wi for xi, wi in zip(x, w))

# Vectorized (Preferred)
dot_vec = np.dot(x, w)
```

### 2.5 Common Mistakes
- **Unintended Array Copies**: Calling `.reshape()` can sometimes create a copy if array memory is non-contiguous (e.g., after transposition). Use `.contiguous()` in PyTorch or `np.ascontiguousarray()` when needed.
- **Axis Confusion in Reductions**: `np.mean(X, axis=0)` reduces across rows (computing column-wise means for features), while `axis=1` reduces across columns (row-wise means).

---

## 3. Pandas: Tabular Data Manipulation

### 3.1 Intuition
Machine learning pipelines frequently ingest tabular records (CSV, Parquet, SQL). Pandas provides column-oriented abstractions (`DataFrame` and `Series`) built on top of NumPy arrays, supporting index alignment, relational joins, and automated aggregations.

### 3.2 Core Operations & Idioms
- **Vectorized Filtering**: Use boolean masks instead of `df.apply()` or iterrows.
- **Handling Missing Values**: Distinguish between imputation (mean, median, forward fill) and indicator variables for missingness:
```python
import pandas as pd

# Downcasting numeric columns to reduce memory by up to 70%
def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
```

### 3.3 Common Mistakes
- **Iterating with `.iterrows()`**: Extremely slow because it converts each row into a new `Series` object. Use vectorized operations or `.itertuples()` for 100x speedups.
- **`SettingWithCopyWarning`**: Caused by chained indexing like `df['col'][mask] = val`. Always use `df.loc[mask, 'col'] = val`.

---

## 4. Matplotlib & Data Visualization

### 4.1 Intuition
Diagnostic visualization is essential for spotting dataset skew, evaluating convergence dynamics, and verifying calibration. Matplotlib's Object-Oriented (OO) API separates Figure containers from Axes subplots, ensuring reproducible layouts.

### 4.2 Production Plotting Pattern
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", lw=2)
ax.plot(epochs, val_losses, label="Val Loss", color="#d62728", ls="--", lw=2)
ax.set_title("Training Loss Convergence", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
```

### 4.3 Common Mistakes
- Relying entirely on stateful `plt.plot()` across multi-figure workflows, which leads to overlapping axes and title collisions.

---

## 5. Functions, Classes, and Useful ML Patterns

### 5.1 Object-Oriented Estimator Pattern
Follow the standard Scikit-Learn / PyTorch design paradigm:
- Initialization in `__init__`: hyperparameter assignment only, no data fitting.
- `fit(X, y)`: computes and stores learned parameters with a trailing underscore (e.g., `self.mean_`).
- `transform(X)` / `predict(X)`: stateless inference on unseen data.

```python
class FeatureStandardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureStandardizer":
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Estimator must be fitted before transforming data.")
        return (X - self.mean_) / (self.std_ + self.eps)
```

### 5.2 Dataclasses for Hyperparameters
Use `@dataclass(frozen=True)` to create immutable configuration schemas:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    weight_decay: float = 1e-4
```

---

## 6. Defensive Programming & Exception Handling

In machine learning pipelines, runtime crashes after hours of training are costly. Always validate incoming tensor invariants defensively:
- **Shape Checks**: Validate rank and compatibility (`assert X.ndim == 2`).
- **Numerical Anomalies**: Check for `np.isnan(X).any()` or `np.isinf(X).any()`.
- **Condition Numbers**: Check for near-singular matrices before inversion using `np.linalg.cond(A)`.

---

## 7. Virtual Environments & Reproducibility

Isolate dependency trees to avoid system-level version conflicts:
```bash
# Standard Python venv setup
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\Activate.ps1  # Windows

# Pinned installation
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Topic Module Resources
- Interactive Lab: [`notebook.ipynb`](./notebook.ipynb)
- Source Modules: [`code/core_patterns.py`](./code/core_patterns.py), [`code/data_processing.py`](./code/data_processing.py), [`code/visualization.py`](./code/visualization.py)
- Pytest Suite: [`code/test_python_for_ml.py`](./code/test_python_for_ml.py)
- Interview Drill: [`interview.md`](./interview.md)
- Curated Citations: [`references.md`](./references.md)
