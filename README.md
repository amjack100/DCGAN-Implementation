# DCGAN CLI with Examples

<!-- ![image](https://img.shields.io/pypi/v/tf_examples.svg%0A%20%20%20%20%20:target:%20https://pypi.python.org/pypi/tf_examples)

![image](https://img.shields.io/travis/amjack100/tf_examples.svg%0A%20%20%20%20%20:target:%20https://travis-ci.com/amjack100/tf_examples)

![image](https://readthedocs.org/projects/tf-examples/badge/?version=latest%0A%20%20%20%20%20:target:%20https://tf-examples.readthedocs.io/en/latest/?badge=latest%0A%20%20%20%20%20:alt:%20Documentation%20Status) -->

Train simple DCGAN models from the CLI while experimenting with
different hyperparameters

- Free software: MIT license
- Documentation: <https://tf-examples.readthedocs.io>.

<p align="center">
  <img src="docs/result-26-feb-2021.gif" alt="animated" />
</p>

Frame by frame animation of training the celebA dataset for 100 epochs

## Installation

```bash
git clone https://github.com/amjack100/DCGAN-Implementation.git
cd ./dcgan
poetry install
poetry run dcgan --help
```

<!-- ![](result-26-feb-2021.gif) -->

Here are some typical trends to look for in a successful GAN training session. These graphs map the discriminator return value when evaluating fake (produced by the generator) and real entities. The different colors represent varying image resolutions of the CelebA dataset (ranging from 4x4 to the original 32x32).

<p align="center">
  <img src="docs/fake_eval.png" alt="animated" />
</p>
<p align="center">
  <img src="docs/real_eval.png" alt="animated" />
</p>

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
