import os
import glob
import time
import random
import argparse

from multiprocessing import Process, Pool

from experiments.event import produce_report_csv
from configs.configs import PipelineConfigs as Configs
# training import 
from experiments.detection import detect_main


def run(process=8):
    # div channels work
    # TODO : should seprate videos not channels
    process_channels_files = glob.glob(os.path.join(Configs.video_path, '*/*.mp4'))
    for i in range(0, len(process_channels_files), process):
        CH_list = [(ch_file, os.path.join(Configs.report_path, ch_file.split('/')[-2])) for ch_file in process_channels_files[i: i + process]]

        pool = Pool()
        # detect
        pool_outputs = pool.map(detect_main, CH_list)  # need wait all process finish
    
    print(pool_outputs)


if __name__ == '__main__':
    #### os cpu count ####
    run(process=1)
    #####################

    produce_report_csv(Configs.report_path, save_path=Configs.report_save_path, save_name=Configs.report_name)