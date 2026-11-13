# Entropy

Entropy is our from-scratch neural-network project. The first model learns a
small puzzle called XOR. This keeps the examples tiny enough to understand and
lets us check the calculations by hand.

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

## Project map

- `neural_network.py` — Entropy's small feedforward network.
- `src/nn/neuron.py` — one artificial neuron with identity and sigmoid output.
- `src/nn/activations.py` — common activations and derivative functions.
- `src/nn/layers.py` — configurable fully connected layers.
- `src/nn/network.py` — feedforward network with multiple hidden layers.
- `src/nn/losses.py` — regression and classification loss functions.
- `src/nn/backprop.py` and `src/nn/gradcheck.py` — learning gradients and
  checks against numerical approximations.
- `src/nn/optimizers.py` and `src/nn/trainer.py` — parameter updates, batches,
  validation, and resumable training.
- `src/nn/linear_algebra.py` — small vector and matrix operations.
- `src/nn/calculus.py` — sigmoid, loss, and derivative examples.
- `src/nn/probability.py` — probability, statistics, entropy, and sampling.
- `src/nn/numerics.py` — stable softmax, log-sum-exp, and precision examples.
- `src/nn/optimization.py` — gradient descent and learning-rate examples.
- `tests/` — hand-calculated neuron checks and experiment acceptance checks.
- `configs/` — fixed training settings and seed.
- `experiments/` — runnable training and learning experiments.
- `docs/checkpoints/` — explanations and evidence for each learning step.
- `reports/` — measured results and the saved plot.

This is a small binary classifier, not a general-purpose assistant. We will
only add complexity after each earlier step is understood and checked.
