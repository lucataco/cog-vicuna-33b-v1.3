#!/usr/bin/env python

import os
import sys
import torch
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import MODEL_NAME, MODEL_CACHE

if os.path.exists(MODEL_CACHE):
    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=MODEL_CACHE,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=MODEL_CACHE,
)