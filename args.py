# -*- coding: utf-8 -*-
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--use_gpu", type=int, default=1)
parser.add_argument("--image_dir", type=str, default="../images")
parser.add_argument("--model_dir", type=str, default="model")
parser.add_argument("--vis_dir", type=str, default="visualization")
args = parser.parse_args()