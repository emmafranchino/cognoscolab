# Example usage in shell

## JSON list

```shell
python3 scripts/run_pipelines.py "'matematica','ansia','stress','depressione']"
```

## CSV

```shell
python3 scripts/run_pipelines.py "matematica,ansia,stress,depressione"
```


## How to run multiple pipelines with one single command and a list of keywords

My name is GitHub Copilot.

You can do it in one shell command by setting RUN_NOTEBOOKS and passing the keywords argument.

Examples (run from the repo root):

- Explicit multiple notebooks (colon-separated):
````bash
RUN_NOTEBOOKS="scripts/bfmns_high_anxiety_highschool_students_gpt4o.ipynb:scripts/bfmns_filtered_highschool_south_2019.ipynb" python3 run_pipelines.py "matematica,ansia,stress,depressione"
````

- Using a glob:
````bash
RUN_NOTEBOOKS="scripts/bfmns_*.ipynb:scripts/tfmns_*.ipynb" python3 run_pipelines.py "['matematica','ansia','stress','depressione']"
````

- Newline-separated list:
````bash
RUN_NOTEBOOKS=$'scripts/a.ipynb\nscripts/b.ipynb' python3 run_pipelines.py "matematica,ansia"
````

Notes:
- The keywords argument accepts CSV (matematica,ansia,…) or a Python/JSON-like list (['matematica','ansia',…]).
- Each targeted notebook must contain a single line like: keywords = [...] in a code cell (the script replaces that line).



---

# How to run with also the minimun association occurrences parameter

## Examples (Linux):

### Using a positional int:

```bash
RUN_NOTEBOOKS="scripts/bfmns_*.ipynb" python3 run_pipelines.py "['matematica','ansia']" 3
```
### Using an env var:

```bash
RUN_NOTEBOOKS="scripts/bfmns_*.ipynb" MIN_ASSOC=3 python3 run_pipelines.py "matematica,ansia"
```

This leaves notebooks unchanged if MIN_ASSOC is not provided.