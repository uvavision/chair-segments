#!/bin/bash
mkdir data
mkdir results
mkdir logs

CHAIRSEGMENT=https://www.cs.virginia.edu/~lp2rv/Datasets/Chair064.zip

echo "Downloading ChairSegment dataset ..."
wget $CHAIRSEGMENT -O Chair064.zip
echo "Unzipping..."
unzip -q Chair064.zip -d data/


rm Chair064.zip