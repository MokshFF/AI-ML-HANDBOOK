name: Bug Report
description: Report a bug, code error, or broken link
title: "[BUG]: "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: Thank you for reporting an issue in `ai-ml-handbook`!
  - type: input
    id: location
    attributes:
      label: File / Topic Location
      description: Where is the bug located? (e.g., `01-machine-learning/supervised/notebook.ipynb`)
      placeholder: path/to/file
    validations:
      required: true
  - type: textarea
    id: description
    attributes:
      label: Description of the Bug
      description: A clear and concise description of what the error is.
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps To Reproduce
      description: How can someone reproduce this behavior?
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What did you expect to happen instead?
    validations:
      required: true
