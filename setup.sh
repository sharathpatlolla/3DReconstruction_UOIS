#!/bin/bash

mkdir -p dependencies/PointConv dependencies/uois3d

git clone https://github.com/martinmatak/PointConv.git
mv PointConv dependencies/

git clone https://github.com/chrisdxie/uois.git
mv uois uois3d
mv uois3d dependencies/
