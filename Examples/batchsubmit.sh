#!/bin/bash

if [ $# -ne 1 ] ; then
  echo "[ERROR] USAGE: ./$0 <python script>"
fi
bsub -q 1nh -J IML-challenge << EOF
source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.sh
cd $PWD
python $1
EOF

