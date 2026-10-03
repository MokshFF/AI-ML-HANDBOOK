# Python for Machine Learning: Technical Interview Question Bank

A curated compendium of conceptual, systems, and coding interview questions focusing on Python runtime internals, NumPy memory mechanics, Pandas performance, and ML engineering patterns.

---

## 1. Runtime Architecture & Memory Internals

### Q1: How does NumPy achieve orders of magnitude speedups compared to native Python lists?
- **Answer Outline**:
  1. **Contiguous Memory Allocation**: Python lists store pointers to heap-allocated PyObject instances scattered throughout memory. NumPy allocates a single continuous block of bytes in memory.
  2. **Cache Locality**: Sequential contiguous access maximizes CPU L1/L2 data cache line hits and enables hardware pre-fetching.
  3. **SIMD Vectorization**: Compilers vectorize operations using CPU instructions (AVX-512, NEON) to process multiple floats (e.g., 8 to 16 floats) per clock cycle.
  4. **Avoidance of Type Boxing**: Python dynamically resolves object types and methods at each iteration; NumPy uses fixed homogeneous dtypes without per-element dispatch overhead.

### Q2: What is the Python GIL (Global Interpreter Lock), and how does it impact multi-threaded machine learning workflows?
- **Answer Outline**:
  - The GIL is a mutual-exclusion lock preventing multiple native threads from executing CPython bytecode simultaneously, protecting reference counts from race conditions.
  - **Impact on ML**: Heavy numerical libraries (NumPy, PyTorch, SciPy, OpenCV) explicitly release the GIL before entering C/C++/CUDA kernels, allowing true multi-core parallel computation.
  - For pure Python preprocessing (tokenization, string manipulation), developers use multiprocessing or ProcessPoolExecutor to bypass the GIL.

---

## 2. Memory Layout: Strides, Views, and Copies

### Q3: What is the difference between a NumPy `view` and a `copy`, and when does slicing create which?
- **Answer Outline**:
  - A **view** is a new array metadata wrapper (`ndarray` header) pointing to the same underlying data buffer, with adjusted shape and strides.
  - Slicing (`arr[1:5]`) creates a view. Changing values in the slice mutates the original array.
  - Boolean masking (`arr[arr > 0]`) and fancy integer indexing (`arr[[0, 2, 4]]`) create **copies**, allocating new memory buffers.
  - Check with `b.base is a` to verify if `b` is a view of `a`.

### Q4: Explain how NumPy strides work. How can you transpose a 2D matrix in $O(1)$ time?
- **Answer Outline**:
  - Strides represent the number of bytes to step in each dimension when advancing by 1 index.
  - For a row-major (C-contiguous) matrix of `float64` with shape $(M, N)$, strides are $(N \times 8, 8)$.
  - Transposition simply swaps the shape tuple $(N, M)$ and the stride tuple $(8, N \times 8)$ without moving any bytes in memory, making it an $O(1)$ operation.

---

## 3. Data Pipelines & High-Performance Pandas

### Q5: Why is `df.iterrows()` considered an anti-pattern in high-throughput data pipelines?
- **Answer Outline**:
  - `iterrows()` generates a new Pandas `Series` for each row, incurring massive Python object creation and index alignment overhead.
  - **Hierarchy of Alternatives**:
    1. Fully vectorized Pandas/NumPy operations (Fastest, ~1000x faster).
    2. List comprehension with `zip(df['col1'], df['col2'])`.
    3. `df.itertuples()` which yields lightweight namedtuples without Series conversion overhead (~50x-100x faster).

### Q6: How do you prevent data leakage during feature preprocessing?
- **Answer Outline**:
  - Preprocessing parameters (e.g., mean and standard deviation for standardization, imputer statistics, target encoding means) must be computed **strictly on the training split**.
  - Call `.fit(X_train)` or `.fit_transform(X_train)`, and only `.transform(X_val)` or `.transform(X_test)`.
  - Fitting transformations on the entire dataset prior to splitting leaks target distribution and variance information into the training phase.

---

## 4. Coding Challenge: Streaming Mini-Batch Loader

### Task
Implement a generator function that takes features `X`, targets `y`, `batch_size`, and a boolean `shuffle`. It must stream batches without duplicating dataset memory.

```python
import numpy as np
from typing import Iterator

def generate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for start_idx in range(0, n, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]
```
