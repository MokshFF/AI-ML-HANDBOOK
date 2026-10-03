name: Topic Proposal
description: Propose a new educational topic or sub-module
title: "[TOPIC]: "
labels: ["enhancement", "topic-proposal"]
body:
  - type: markdown
    attributes:
      value: Propose expanding `ai-ml-handbook` with a new topic module.
  - type: input
    id: module
    attributes:
      label: Target Module
      description: Which major module does this belong to? (e.g., `06-generative-ai`)
      placeholder: "06-generative-ai"
    validations:
      required: true
  - type: input
    id: topic-name
    attributes:
      label: Proposed Topic Name
      description: Lowercase hyphenated folder name (e.g., `speculative-decoding`)
      placeholder: "speculative-decoding"
    validations:
      required: true
  - type: textarea
    id: rationale
    attributes:
      label: Topic Rationale & Objectives
      description: Why should this topic be included? What are the key concepts covered?
    validations:
      required: true
