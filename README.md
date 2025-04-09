# Catan AI

## Installation

### MacOS/Linux

Create a virtual environment and activate it:

```bash
python -m venv .venv
. .venv/bin/activate
```

Install requirements and pre-commit hooks:

```bash
pip install .
pre-commit install
```

### Windows

```Powershell
python -m venv .venv
. .\venv\Scripts\Activate.ps1
```

If this above command produces an error related to execution policy
```Powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
After setting the policy, retry activation:
```Powershell
. .\venv\Scripts\Activate.ps1
```

## Visualizing Games

Install the `catanatron_server` package:

```bash
pip install -e 'git+https://github.com/bcollazo/catanatron#egg=catanatron_server&subdirectory=catanatron_server'
```
