#! /bin/bash
if [ ! -d "debugging/exercise2" ]
then
    echo "Please cd to the root of your pagayapro repo and run the script again"
    exit
fi
if [ ! -d "~/debugging_ex2_venv" ]
then
    /usr/bin/python3 -m venv ~/debugging_ex2_venv
fi
echo "Creating venv"
source ~/debugging_ex2_venv/bin/activate
echo "Installing dependencies"
python -m pip install -U pip
python -m pip install scikit-learn==0.23.0 pandas jupyterlab matplotlib
echo "Patching sklearn with bugs"
rm -r /tmp/sklearn
cp debugging/exercise2/sklearn.zip /tmp/ && cd /tmp && unzip sklearn.zip && cp -r sklearn ~/debugging_ex2_venv/lib/python3.8/site-packages
echo "DONE"
