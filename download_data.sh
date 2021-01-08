#!/bin/bash
mkdir data
mkdir results
mkdir logs

CHAIRSEGMENT64=https://www.cs.virginia.edu/~lp2rv/Datasets/Chair064.zip
#CHAIRSEGMENT128=https://www.cs.virginia.edu/~lp2rv/Datasets/Chair128.zip

echo "Downloading ChairSegment dataset ..."
wget $CHAIRSEGMENT64 -O Chair064.zip
#wget $CHAIRSEGMENT128 -O Chair128.zip

echo "Unzipping..."
unzip -q Chair064.zip -d data/
#unzip -q Chair128.zip -d data/

rm Chair064.zip
#rm Chair128.zip