Installation
============

This tutorial will walk you through the process of setting up an environment
to run the BayesTME pipeline.

Option 1 (Recommended): Using Docker
------------------------------------

BayesTME uses several python packages with C extensions,
so the easiest way to get started is using the up to date
docker image we maintain on docker hub.

.. code::

    docker pull jeffquinnmsk/bayestme:latest

Option 2: Install Using pip
---------------------------

The ``bayestme`` package can be installed directly from github:

.. code::

    pip install git+https://github.com/tansey-lab/bayestme
