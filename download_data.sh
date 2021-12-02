#!/bin/bash
mkdir data
mkdir results
mkdir logs

#CHAIRSEGMENT32=https://www.cs.rice.edu/~vo9/chair-segments/Chair032.zip
CHAIRSEGMENT64=https://www.cs.rice.edu/~vo9/chair-segments/Chair064.zip
#CHAIRSEGMENT128=https://www.cs.virginia.edu/~lp2rv/Datasets/Chair128.zip

echo "Downloading ChairSegment dataset ..."
wget $CHAIRSEGMENT64 -O Chair064.zip
#wget $CHAIRSEGMENT32 -O Chair032.zip
#wget $CHAIRSEGMENT128 -O Chair128.zip

echo "Unzipping..."
unzip -q Chair064.zip -d data/
#unzip -q Chair032.zip -d data/
#unzip -q Chair128.zip -d data/

rm Chair064.zip
#rm Chair032.zip
#rm Chair128.zip
