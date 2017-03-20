# FAQ and Known Issues

## pip gets stuck
If you try to install packages in your user space with pip, it gets stuck at the end of the execution with LCG88. However, the packages appear to be installed correctly.
As a temporary workaround, if you need to install packages, run the install from the terminal (not from a notebook) and interupt pip with C-c

## import ROOT gets stuck
This is a known bug in ROOT.py. If this happens, restart the jupyter kernel and try again.