#!/bin/bash

mkdir SVHN_data

curl http://ufldl.stanford.edu/housenumbers/train_32x32.mat --output SVHN_data/train_32x32.mat
curl http://ufldl.stanford.edu/housenumbers/extra_32x32.mat --output SVHN_data/extra_32x32.mat
curl http://ufldl.stanford.edu/housenumbers/test_32x32.mat --output SVHN_data/test_32x32.mat

