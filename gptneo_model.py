import sys
!{sys.executable} -m pip install git+https://github.com/huggingface/transformers

from transformers import pipeline

generator("My day", do_sample=True, min_length=100)