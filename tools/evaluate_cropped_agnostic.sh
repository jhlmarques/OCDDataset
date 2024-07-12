#!/bin/bash
dir="configs/OCD-Cropped"
cd mmrotate-1.x
out_file="../detection_outputs/agnostic.out"

rm -f $out_file


for config_file in "$dir"/*; do
    config_basename="$(basename -- $config_file)"
    config_noext="${config_basename%.*}"
    checkpoint=work_dirs/"${config_noext}"/epoch_72.pth
    echo $config_noext >> $out_file
    echo ---------------- >> $out_file
    #echo "TWO CLASSES: " >> $out_file
    python tools/test.py $config_file $checkpoint 2>&1 | tail -1 >> $out_file
    #echo "CLASS AGNOSTIC: " >> $out_file
    config_options="--cfg-options test_evaluator.type='OCDMetric' --cfg-options test_evaluator.path_test_images='/home/jose/Programas/OCDDataset/mmrotate-1.x/data/OCD/test/images'"
    python tools/test.py $config_file $checkpoint $config_options 2>&1 | tail -1 >> $out_file
    echo ---------------- >> $out_file
done