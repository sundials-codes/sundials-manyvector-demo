#!/bin/bash

../../../scripts/parse_timing_data.py \
    imex/stdout-719209.txt \
    imex/stdout-719210.txt \
    imex/stdout-719211.txt \
    imex/stdout-719212.txt \
    mri/stdout-719204.txt \
    mri/stdout-719205.txt \
    mri/stdout-719206.txt \
    mri/stdout-719207.txt \
    --printtime
