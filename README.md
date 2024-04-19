# Python-Project-Template

This is a template repository. Please initialize your python project using this template.

### 1. Verify and Update Python Version
Ensure that the correct version of Python is installed on your local system. Update the Python version in the following configuration files:
- `.github/workflows/*.yml`
- `pyproject.toml`

### 2. Project Package Name
Replace `project_name` with your actual project package name, which should include the `src` directory.

### 3. Setting Up CI/CD for Release
To enable Continuous Integration and Continuous Deployment (CI/CD) for releasing your software, follow these steps:

- Configure `CI_CD_TOKEN` as a secret variable through the Actions Settings in your repository.
- To release, add a tag to your repository by executing the following commands:
```
git tag -a v0.0.1 -m "Release version 0.0.1"
git push origin v0.0.1
```
## Installing and Using the Package

You can install the package locally from the top-level directory:
```python
python -m pip install --upgrade pip build
python -m build
python -m pip install .
```

Distribute
If you want others to install your package via pip directly, you can upload it to PyPI. This requires an account on PyPI and then you can upload using Twine:
```
python -m pip install --upgrade twine
twine upload dist/*
```

## Ruff Usage
Here is how you might proceed with checking for issues:
```
ruff check .
```
If you need to fix issues automatically and if ruff supports it, you would typically see an option like `--fix` mentioned in the documentation or help command:
```
ruff check --fix .
```