name: Feature Request
description: Suggest an enhancement to repository infrastructure or tooling
title: "[ENHANCEMENT]: "
labels: ["enhancement"]
body:
  - type: textarea
    id: description
    attributes:
      label: Proposal Description
      description: What improvement would you like to see?
    validations:
      required: true
  - type: textarea
    id: motivation
    attributes:
      label: Motivation & Use Case
      description: Why is this feature beneficial to readers and contributors?
    validations:
      required: true
