#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: willc
"""

import glob
import os
import ntpath
import random

if __name__ == "__main__":
    total_files_idx = list(range(0, 7481))
    random.shuffle(total_files_idx)


    val_idx = sorted(total_files_idx[:1000])
    train_idx = sorted(total_files_idx[1001:])

    val_idx = [str(x) for x in val_idx]
    train_idx = [str(x) for x in train_idx]

    val_id_text = "\n".join(val_idx)
    train_idx_text = "\n".join(train_idx)


    with open("val_id.txt", 'w') as fout:
        fout.write(val_id_text)

    with open("train_id.txt", 'w') as fout:
        fout.write(train_idx_text)