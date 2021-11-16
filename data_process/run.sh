#!/usr/bin/env sh

# USE YOUR OWN PYTHON ENV.

# amazon dataset.
python public_data_gen_torch.py

# collected dataset.
python business_process.py --rate 0.1
