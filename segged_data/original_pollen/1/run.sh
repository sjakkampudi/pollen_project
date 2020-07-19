#!/bin/bash

# Rename all *.png.dat to *.dat
for f in *.dat.png; do 
    mv -- "$f" "${f%.dat.png}.png"
done
