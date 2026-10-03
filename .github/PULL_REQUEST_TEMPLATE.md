## Description
<!-- Provide a clear, concise summary of the changes introduced in this PR. -->

## Type of Change
- [ ] New topic added (following `_templates/topic-template/`)
- [ ] Code correction or algorithm bug fix
- [ ] Documentation improvement or formula correction
- [ ] CI/CD or repository infrastructure enhancement

## Topic Verification Checklist (For educational content)
- [ ] Follows the standard 5-file architecture (`README.md`, `notebook.ipynb`, `code/`, `interview.md`, `references.md`)
- [ ] Notebook executes cleanly from top to bottom
- [ ] Notebook outputs are cleared of heavy images/tensors
- [ ] No hardcoded API keys or environment secrets
- [ ] Code formatted with PEP 8 standards
- [ ] Relative links verified

## Validation Commands Run
```bash
python scripts/validate_repo.py
ruff check .
```
