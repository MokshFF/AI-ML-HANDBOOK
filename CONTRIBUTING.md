# Contributing to `ai-ml-handbook`

Thank you for your interest in contributing to the **`ai-ml-handbook`**! This repository is designed to be an open, rigorous, and continuously evolving resource for the machine learning and AI community.

To maintain professional engineering standards, clarity, and consistency across all modules, please review these guidelines before submitting an issue or pull request.

---

## 1. Code of Conduct & Contribution Philosophy

- **Pedagogical Rigor**: Explanations must be technically accurate, mathematically grounded, and free of exaggerated hype.
- **Reproducibility**: All code must execute cleanly from scratch with pinned dependencies.
- **Consistency**: Follow the established directory conventions and templates.

---

## 2. Topic Directory Standard

Every educational topic directory must adhere to the standard five-component structure established in [`_templates/topic-template/`](./_templates/topic-template/):

```
topic-slug/
├── README.md         # Topic overview, learning objectives, and concept matrix
├── notebook.ipynb    # Clean, runnable Jupyter notebook
├── code/             # Standalone Python scripts & reusable modules
│   └── .gitkeep
├── interview.md      # Theory questions, trade-offs, and coding drills
└── references.md     # Seminal papers, textbooks, and documentation links
```

When proposing a new topic:
1. Copy `_templates/topic-template/` into the appropriate module directory.
2. Rename the directory using lowercase hyphenated naming (e.g., `feature-engineering`).
3. Fill out all sections without leaving empty placeholder strings or broken links.

---

## 3. Documentation Standards

- **Clean Markdown**: Use GitHub Flavored Markdown (GFM). Avoid raw HTML unless strictly necessary for rendering.
- **Mathematical Equations**: Use standard LaTeX notation enclosed in `$...$` for inline math and `$$...$$` on its own line for display equations.
- **Diagrams**: Use Mermaid diagrams (`mermaid` code blocks) for flowcharts, sequences, and architecture blueprints.
- **Relative Linking**: Always use relative paths for repository-internal links (e.g., `[Linear Algebra](./00-prerequisites/math-linear-algebra/)`). Never hardcode `localhost` or absolute machine paths.

---

## 4. Coding Standards

- **Python Version**: Python 3.10+ compatible.
- **Style Guide**: Conform to PEP 8. Format code using `black` (88 character line limit) and check with `ruff`.
- **Type Annotations**: Provide explicit type hints for all public function signatures:
  ```python
  def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
      ...
  ```
- **Docstrings**: Provide clear Google-style or NumPy-style docstrings describing parameters, return types, and exceptions.
- **Vectorization**: Favor vectorized NumPy/PyTorch operations over explicit Python loops for mathematical computation.

---

## 5. Notebook Standards

- **Execution State**: Notebooks committed to the repository must have cleared outputs or cleanly executed minimal runs (no hundreds of megabytes of binary image outputs).
- **Run Order**: Ensure the notebook can be executed sequentially from Top to Bottom using **Kernel -> Restart and Run All**.
- **No Hardcoded Credentials**: Never commit API keys, tokens, or environment credentials. Use `os.getenv(...)` or standard `.env` configuration.
- **Lightweight Dependencies**: Avoid pulling massive datasets inside notebooks. Use synthetic data generators (`sklearn.datasets.make_classification`) or small standard benchmark subsets.

---

## 6. Citation & Attribution Requirements

- All algorithm implementations adapted from research papers or third-party open-source projects must include appropriate attribution.
- Citations in `references.md` should include:
  - Title of the paper/book
  - Authors
  - Year of publication
  - Link to arXiv, DOI, or official project page

---

## 7. Pull Request & Issue Process

### Reporting Issues or Proposing Topics
1. Check existing [Issues](https://github.com/avars/ai-ml-handbook/issues) to avoid duplicate proposals.
2. Select the appropriate issue template:
   - **Bug Report**: For code errors, broken links, or inaccurate equations.
   - **Topic Proposal**: For proposing a new educational module or subtopic.
   - **Feature / Improvement**: For structural or infrastructural enhancements.

### Submitting a Pull Request (PR)
1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/topic-name
   ```
2. Validate repository structure locally:
   ```bash
   python scripts/validate_repo.py
   ```
3. Run linting checks:
   ```bash
   ruff check .
   ```
4. Commit your changes with clear, semantic commit messages (e.g., `feat(dl): add transformer attention lab`).
5. Open a Pull Request referencing any related issues and complete the PR checklist.
