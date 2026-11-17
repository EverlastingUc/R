# Entropy

Entropy is our from-scratch neural-network research project. The first model
learns a small puzzle called XOR. This keeps the examples tiny enough to
understand and lets us check the calculations by hand.

## How we train it

1. Give Entropy examples and the correct answers.
2. Entropy makes a guess using its current weights. Weights are adjustable
   numbers inside the network.
3. Compare its guesses with the correct answers. The average mistake is called
   the loss.
4. Backpropagation works out how each weight contributed to the mistake.
5. Gradient descent nudges the weights to make the loss smaller.
6. Repeat with the examples. Then check the model on examples it did not train
   on.

For XOR, the four input pairs and answers are:

| First input | Second input | Answer |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The first training example shows this loop in code. The separate experiment
compares learning a real pattern with memorizing random answers. It shows why a
high score on training examples alone is not enough.

## Set up the project

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Run the neuron and experiment checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the training experiment and save a fresh plot and metrics record:

```powershell
.\.venv\Scripts\python.exe -m experiments.memorization_vs_generalization
```

Train Entropy with gradient descent on a tiny generated regression task:

```powershell
.\.venv\Scripts\python.exe experiments\cp18_gradient_descent.py
```

Run the full training pipeline with small batches, shuffled examples,
validation checks, and save/restore:

```powershell
.\.venv\Scripts\python.exe experiments\cp20_training_pipeline.py
```

The full pipeline uses generated examples, so it needs no download. It saves a
new numbered training report and model checkpoint under `reports/` each time.
The settings and random seeds are in `configs/cp20_training_pipeline.json`.

## Proton-I: Gluons (Stage 1)

Gluons is Proton-I's first deliberately limited assistant. It searches the web,
selects a short relevant result snippet, and prints one plain-text paragraph
with a source URL. It does not use Markdown. JavaScript and framework requests
are outside this stage.

Gluons 0.1 is a retrieval-based baseline, not a trained generative language
model. The existing Entropy neural-network modules and checkpoints remain
unchanged; no language-answering weights are claimed. The search adapter uses
DuckDuckGo's HTML results page as a best-effort provider, so it can be affected
by provider rate limits or markup changes.

Run Gluons from PowerShell after installing the project:

```powershell
python -m proton_i.gluons "What is CSS Flexbox?"
```

Or use the installed command:

```powershell
gluons "What is semantic HTML?"
```

See [the Gluons stage guide](docs/proton_i/GLUONS.md) for its exact scope,
limitations, and tests.

## Project map

- `neural_network.py` — Entropy's small binary network.
- `src/nn/` — NumPy neurons, layers, feedforward network, losses, backprop,
  optimizers, trainer, and numerical utilities.
- `src/proton_i/gluons/` — Proton-I Stage 1 plain-text web assistant.
- `tests/` — neural-network and Gluons tests.
- `configs/` — fixed training settings and seeds.
- `experiments/` — runnable training and learning experiments.
- `docs/checkpoints/` — explanations and evidence for Entropy checkpoints.
- `docs/proton_i/` — Proton-I stage specifications and guides.
- `reports/` — measured experiment results and saved checkpoints.

Entropy's current research models are small experimental networks, not
general-purpose assistants. Gluons is a separate retrieval-based prototype;
its capabilities are limited to the behavior described above.
