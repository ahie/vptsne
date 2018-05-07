#!/usr/bin/env python

import setuptools

setuptools.setup(
  name="vptsne",
  version="0.0.1",
  description="Tensorflow ops for computing the t-SNE loss and gradient.",
  author="Aleksi Hietanen",
  maintainer="Aleksi Hietanen",
  maintainer_email="aleksi.s.hietanen@gmail.com",
  install_requires=[
    "numpy"
  ],
  packages=["vptsne"],
  package_data={ "vptsne": ["tsne_loss.so"] },
  url="https://github.com/ahie/vptsne",
  license="MIT")

