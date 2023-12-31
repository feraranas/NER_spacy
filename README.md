# 🪐 spaCy Project: Novel nlp component to do relation extraction.

We implement a spaCy component with a custom Machine Learning model, how to train it with and without a transformer, and how to apply it on an evaluation dataset.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `data` | Parse the gold-standard annotations from the Prodigy annotations. |
| `train_cpu` | Train the REL model on the CPU and evaluate on the dev corpus. |
| `train_gpu` | Train the REL model with a Transformer on a GPU and evaluate on the dev corpus. |
| `evaluate` | Apply the best model to new, unseen text, and measure accuracy at different thresholds. |
| `clean` | Remove intermediate files to start data preparation and training from a clean slate. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `data` &rarr; `train_cpu` &rarr; `evaluate` |
| `all_gpu` | `data` &rarr; `train_gpu` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/annotations.jsonl`](assets/annotations.jsonl) | Local | Gold-standard REL annotations created with Prodigy |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

### Visualizer
Visualize the proyect via streamlit: https://skill-taxonomy.streamlit.app/
