# Free-form flows: Make Any Architecture a Normalizing Flow 

This is the official `PyTorch` implementation of [our preprint](http://arxiv.org/abs/2310.16624):

```bibtex
@article{sorrenson2023maximum,
    title = {Maximum Likelihood Training of Autoencoders},
    author = {Draxler, Felix and Sorrenson, Peter and Rousselot, Armand and Zimmermann, Lea and Köthe, Ullrich},
    journal = {arXiv preprint arXiv:2310.16624},
    year = {2023}
}
```

The code is also compatible with Free-form injective flows (FIF), presented in [this preprint](http://arxiv.org/abs/2306.01843):

```bibtex
@article{sorrenson2023maximum,
    title = {Lifting architectural constraints in injective normalizing flows},
    author = {Sorrenson, Peter and Draxler, Felix and Rousselot, Armand and Hummerich, Sander and Zimmermann, Lea and Köthe, Ullrich},
    journal = {arXiv preprint arXiv:2306.01843},
    year = {2023}
}
```


## Installation

Options:

1. Install via `pip` and use our package.
2. Copy the loss script with `PyTorch` as the only dependency to train in your own setup.

### Install via pip

The following will install our package along with all of its dependencies:

```bash
git clone https://github.com/vislearn/FFF.git
cd FFF
pip install -r requirements.txt
pip install .
```

In the last line, use `pip install -e .` if you want to edit the code.

Then you can import the package via

```python
import fff
```

### Copy `fff/loss.py` into your project

If you do not want to add our `fff` package as a dependency,
but still want to use the FFF loss function,
you can copy the `fff/loss.py` file into your own project.
It does not have any dependencies on the rest of the repo.


## Basic usage

### Train your architecture 

```python
import torch
import fff.loss as loss


class FreeFormFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(...)
        self.decoder = torch.nn.Sequential(...)


model = FreeFormFlow()
optim = ...
data_loader = ...
n_epochs = ...
beta = ...

for epoch in range(n_epochs):
    for batch in data_loader:
        optim.zero_grad()
        loss = loss.fff_loss(batch, model.encoder, model.decoder, beta)
        loss.backward()
        optim.step()
```


### Build models based on our framework

Our training framework is built on https://github.com/LarsKue/lightning-trainable based on PyTorch Lightning. There is no `main.py`, but
you can train all our models via the `lightning_trainable.launcher.fit` module.
For example, to train the Boltzmann generator on DW4:
```bash
python -m lightning_trainable.launcher.fit configs/dw4.yaml --name '{data_set[name]}'
```

This will create a new directory `lightning_logs/dw4/`. You can monitor the run via `tensorboard`:
```bash
tensorboard --logdir lightning_logs
```

When training has finished, you can import the model via
```python
import fff

model = fff.model.FreeFormFlow.load_from_checkpoint(
    'lightning_logs/dw4/version_0/checkpoints/last.ckpt'
)
```

If you want to overwrite the default parameters, you can add `key=value`-pairs after the config file:
```bash
python -m lightning_trainable.launcher.fit configs/dw4.yaml batch_size=128 loss_weights.noisy_reconstruction=20 --name '{data_set[name]}'
```

### Known issues

Training with $E(n)$-GNNs is sometimes unstable. This is usually caught with an assertion in a later step and training is stopped.
In almost all cases, training can be stably resumed by passing the `--continue-from [CHECKPOINT]` flag to the training, such as:
```bash
python -m lightning_trainable.launcher.fit configs/dw4.yaml --name '{data_set[name]}' --continue-from lightning_logs/dw4/version_0/checkpoints/last.ckpt
```
This reloads the entire training state (model state, optim state, epoch, etc.) from the checkpoint and continues training from there.
