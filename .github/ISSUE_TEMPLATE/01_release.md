---
name: New release
about: Release checklist
title: Release 2025.X.Y
labels: ''
assignees: '' 
---

## Release checklist

- [ ] Update version in `pyproject.toml`. Use date versioning (e.g. 2025.8.1 for August 1, 2025)
- [ ] Make sure Python version is [reasonably up to date](https://pyreadiness.org/) in:
  - [ ] `pyproject.toml` 
  - [ ] `environment.yaml`
  - [ ] `.github/workflows/pypi.yaml`
- [ ] Make sure dependencies are up to date:
  - [ ] Imports in `src` should be added to `pyproject.toml`
  - [ ] Imports in `scripts`/`notebooks` should be added to `environment.yaml`
  - [ ] Regenerate conda lock file if dependencies are added
- [ ] Regenerate the API documentation (see [Documentation](#documentation))
- [ ] Regenerate output
  - [ ] Save it in a timestamped / versioned directory 
  - [ ] Use [test script](#testing) to verify new output has no unexpected changes
  - [ ] Upload to Zenodo
  - [ ] Upload to CABE
- [ ] Check:
  - [ ] CLI commands still work
  - [ ] Recent notebooks still work
  - [ ] Scripts still work
- [ ] Add new contributors to `CITATION.CFF`
- [ ] [Create GitHub release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) and wait for workflow to finish
- [ ] Verify release on [PyPI](https://pypi.org/project/effort-sharing/) (`pip install effort-sharing`)
- [ ] Verify release on Zenodo
- [ ] Celebrate