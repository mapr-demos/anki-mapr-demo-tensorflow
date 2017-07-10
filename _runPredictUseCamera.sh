#!/bin/bash

useAnki=${1}

if [ "${useAnki}" = "" ]
then
	useAnki="1"
else
	useAnki="0"
fi

python ../TF/predict_camera.py --test_dir=/vagrant/images --model_path=/vagrant/7.sgcsfn-120/model-0.00162638 --poll_frq=1 --use_camera=1 --use_anki=${useAnki}

