#!/bin/bash

for ((counter = 0; counter < 10000; counter += 100))
do
# Generate the next group of synapse image chunks
python 00_gen_data_masks.py $counter
# Copy this group of image chunks to the mb-microns-data bucket
gsutil -m cp -r ../resources/img_chunk_masks_8_8_40 gs://mb-microns-data
# Remove the files in the current local directory
rm ../resources/img_chunk_masks_8_8_40/*
done