Installation
============

This tutorial will walk you through the process of setting up an environment
to run the BayesTME pipeline.

Option 1 (Recommended): Using Docker
------------------------------------

BayesTME uses several python packages with C extensions,
so the easiest way to get started is using the up to date
docker image we maintain on docker hub.

    $ docker pull jeffquinnmsk/bayestme:latest

Option 2: Install Using pip
-------------------------

Installing the package directly will require `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_

The ``bayestme`` package can be installed directly from github:

    $ pip install git+https://github.com/tansey-lab/bayestme

