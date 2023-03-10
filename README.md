# mumate

mumate is an extension of the research I did to satisfy my undergraduate degree requirements: numerical simulations of MnSi thin films (Dalhousie University)

Included is an .mx3 source file to run a simple program using the [MuMax3 Simulator][https://github.com/mumax/3] in order to generate OVF2 files for development.
This file was modified from the benchmark given with mumax3, and stays with a single geometry over multiple time steps.

The main purpose of this repo will be to implement the tools I built to interpret the data and hopefully package them as general purpose python modules.

## Installation
In a new virtualenv
`pip install -r requirements.txt`

## Generate OVF2 files
It is assumed that you have already set up mumax3:
`mumax3 ./bench.mx3`