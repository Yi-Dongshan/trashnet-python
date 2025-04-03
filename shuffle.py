import os
import random

# 参数设置
class Config:
    filename = 'data/one-indexed-files.txt'
    format = 'one-indexed-files-notrash_'
    outputDir = 'data'
    train = 0.70
    val = 0.13
    test = 0.17

opt = Config()

assert opt.filename, "Need a text file consisting of filenames and labels."
assert opt.train != 0, "Must have train examples."
assert opt.val != 0, "Must have val examples."
assert opt.test != 0, "Must have test examples."

def shuffle(opt):
    with open(opt.filename, 'r') as file:
        all_lines = file.readlines()

    random.shuffle(all_lines)

    splits = {
        'train': int(len(all_lines) * opt.train),
        'val': int(len(all_lines) * opt.val),
        'test': len(all_lines) - int(len(all_lines) * opt.train) - int(len(all_lines) * opt.val)
    }

    start = 0
    for split, count in splits.items():
        end = start + count
        split_lines = all_lines[start:end]
        start = end

        output_file = os.path.join(opt.outputDir, f"{opt.format}{split}.txt")
        with open(output_file, 'w') as file:
            file.writelines(split_lines)

shuffle(opt)