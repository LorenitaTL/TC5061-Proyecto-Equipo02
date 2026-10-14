# mi-bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Mi Bici ML

## Setup

The project requires Python 3.12. We use [uv](https://docs.astral.sh/uv/) to create the
virtual environment (install it with `brew install uv` on macOS, or see the
[installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
make create_environment
source .venv/bin/activate
make requirements
```

This installs the dependencies and the [pre-commit](https://pre-commit.com/) git hooks
(`make setup-hooks` installs only the hooks).

Activate the environment with `source .venv/bin/activate` every time you open a new
terminal to work on the project.

### Windows (without `make` or `uv`)

If you don't have `make` or `uv`, create the environment with the standard library's
`venv` and `pip`. Make sure Python 3.12 is installed (`py -3.12 --version`):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
pre-commit install
```

Activate the environment with `.venv\Scripts\activate` every time you open a new
terminal. Run the tests with `python -m pytest tests` instead of `make test`.

### Pre-commit

On every `git commit`, the hooks in `.pre-commit-config.yaml` run on the staged files:

- `ruff` lints (with `--fix`) and formats the code, using the config in `pyproject.toml`.
- Generic checks: trailing whitespace, end of file, YAML/TOML syntax, merge conflict
  markers, private keys, and files larger than 1 MB.
- `pytest` runs the test suite when the commit includes Python files. It uses the
  Python of the active environment, so activate `.venv` before committing.

If a hook modifies files, the commit is aborted. Review the changes, `git add` them, and
commit again. To run the hooks on the whole repo:

```bash
pre-commit run --all-files
```

## Git workflow

We follow [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):
`main` always works, and every change goes through a branch and a Pull Request (PR).

`main` is protected: nobody can push to it directly, and every PR needs one approval
before it can be merged.

### Step by step

```bash
# 1. Start from an up-to-date main
git checkout main
git pull

# 2. Create a branch for your change
git checkout -b feat/peak-hour-features

# 3. Work and commit (as many commits as you need)
git add .
git commit -m "feat: add peak hour features"

# 4. Push the branch
git push -u origin feat/peak-hour-features

# 5. Open the PR
gh pr create --base main --fill
```

Then:

6. Ask a teammate to review the PR.
7. If the reviewer asks for changes, commit them on the same branch and `git push` again.
   The PR updates automatically.
8. Once the PR is approved and the `pre-commit` check is green, merge it with
   **Squash and merge** and delete the branch.
9. Go back to step 1 for your next change.

### Branch names

`type/short-description`, using the same types as the commit messages, in lowercase and
with hyphens instead of spaces:

```
feat/data-ingestion
fix/station-encoding
docs/ml-canvas
```

### Commit messages

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
`type: description`. A hook rejects the commit if the message doesn't match.

| Type       | Use it for                                  |
|------------|---------------------------------------------|
| `feat`     | New functionality                           |
| `fix`      | Bug fix                                     |
| `docs`     | Documentation only                          |
| `test`     | Adding or changing tests                    |
| `refactor` | Code change that doesn't change behavior    |
| `chore`    | Anything else (dependencies, config, etc.)  |

Examples:

```
feat: add peak hour features
fix: handle missing station ids
docs: explain how to download the data
```

### Pull Requests

- **Title:** same format as the commit messages (`type: description`). Because PRs are
  merged with *Squash and merge*, the PR title becomes the commit message in `main`.
- **Description:** what changed, why, and how to test it. Link the related issue if
  there is one (`Closes #<issue number>`).
- **Size:** keep PRs small, focused on a single change. They are easier to review.

### Reviews

Every team member opens at least one PR and reviews at least one PR from a teammate.
To review, go to the PR's **Files changed** tab, leave comments on specific lines if
needed, and click **Review changes**:

- **Approve:** the change is ready to merge.
- **Request changes:** something must be fixed first. Explain what and why.
- **Comment:** questions or suggestions that don't block the merge.

All review conversations must be resolved before merging.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mi_bici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mi_bici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mi_bici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
