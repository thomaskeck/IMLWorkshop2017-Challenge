#!/bin/bash

if [ $# -ne 2 ] ; then
  echo "[ERROR] USAGE: $0 <input notebook file> <output python script>"
  exit 1
fi
echo "[INFO] You need iPython version greater 1.0, ensure that you have"
echo "       activated a recent CVMFS software stack."
echo "[INFO] e.g.:"
echo "[INFO]   source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.sh"

jupyter nbconvert --to python "$1" --output "$2" --output-dir="."
if [ $? -ne 0 ] ; then
  echo "[ERROR] conversion failed"
  exit 1
fi

echo "[INFO] Outcomment iPython magic functions in output code."
sed -i '/get_ipython()/s/^/#/' "$2"

echo "[INFO] Testing occurances of matplotlib's 'plt.show()'"
\grep -e "\w.show()" "$2" > /dev/null
if [ $? -ne 0 ] ; then
  echo "[INFO] Your script seems fine"
else
  echo "[WARNING] Please check! Comment out, or convert to 'plt.savefig(\"plot.png\")'"
  echo "[WARNING] at the following occurances"
  \grep -nHe "\w.show()" "$2"
fi

echo "[INFO] Testing matplotlib setup"
\grep "\<matplotlib\>" $2 > /dev/null
if [ $? -ne 0 ] ; then
  echo "[INFO] no sign of matplotlib found. Your script should be fine"
else
  \grep "\<Agg\>" $2 > /dev/null
  if [ $? -ne 0 ] ; then
    echo "[WARNING] Your matplotlib setup might fail on batch systems"
    echo "[WARNING] consider adding the following lines"
    echo "          import matplotlib"
    echo "          matplotlib.use('Agg')"
    echo "[INFO] see http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined"
  fi
fi
