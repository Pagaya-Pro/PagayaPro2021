#! /bin/bash
if [ ! -d "debugging/exercise2" ]
then
    echo "Please cd to the root of your pagayapro repo and run the script again"
    exit
fi
rm -r /tmp/sklearn
cp debugging/exercise2/sklearn.zip /tmp/ && cd /tmp && unzip sklearn.zip && cp -r sklearn ~/debugging_ex2_venv/lib/python3.8/site-packages
echo "DONE"
