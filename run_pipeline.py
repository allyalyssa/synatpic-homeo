"""Run the whole pipeline end-to-end for one or more datasets.

    python run_pipeline.py                 # dreams (default)
    python run_pipeline.py dreams sleep_edf

Assumes the raw data is already downloaded (`python download.py`). Each stage
reads the previous stage's output from disk, so stages can also be run one at a
time from their own modules.
"""
import sys

import preprocess
import slow_waves
import dissipation
import features
import train
import interpret

RUNNERS = {"dreams": preprocess.run_dreams,
           "dreams_subjects": preprocess.run_dreams_subjects,
           "sleep_edf": preprocess.run_sleep_edf}


def run(dataset):
    print(f"\n########## {dataset} ##########")
    RUNNERS[dataset]()                 # stage 2: preprocess
    slow_waves.run(dataset)            # stage 3
    dissipation.run(dataset)           # stage 4
    features.build(dataset)            # stage 5
    train.cross_validate(dataset)      # stages 6-7
    interpret.summarize(dataset)       # stage 8


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["dreams"]):
        run(ds)
