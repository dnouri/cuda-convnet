This is my fork of the ``cuda-convnet`` convolutional neural network
implementation written by Alex Krizhevsky.

``cuda-convnet`` has quite extensive documentation itself.  Find the
`MAIN DOCUMENTATION HERE <http://code.google.com/p/cuda-convnet/>`_.

===================
Additional features
===================

This document will only describe the small differences between
``cuda-convnet`` as hosted on Google Code and this version.

Dropout
=======

Dropout is a relatively new regularization technique for neural
networks.  See the `Improving neural networks by preventing
co-adaptation of feature detectors <http://arxiv.org/abs/1207.0580>`_
and `Improving Neural Networks with Dropout
<http://www.cs.toronto.edu/~nitish/msc_thesis.pdf‎>`_ papers for
details.

To set a dropout rate for one of our layers, we use the ``dropout``
parameter in our model's ``layer-params`` configuration file.  For
example, we could use dropout for the last layer in the CIFAR example
by modifying the section for the fc10 layer to look like so::

  [fc10]
  epsW=0.001
  epsB=0.002
  # ...
  dropout=0.5

In practice, you'll probably also want to increase the number of
hidden units in that layer (``outputs``), and change the ``wc`` and
``initW`` parameters, too.


CURAND random seeding
=====================

An environment variable ``CONVNET_RANDOM_SEED``, if set, will be used
to set the CURAND library's random seed.  This is important in order
to get reproducable results.


Minor changes
=============

Minor changes include using more Debian/Ubuntu default for the paths
defined in the ``build.sh`` script and ``Makefile``.
