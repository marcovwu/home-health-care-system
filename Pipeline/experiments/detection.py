import os
import sys
import csv
import time
import glob
import shutil

from tqdm import tqdm

from configs.configs import PipelineConfigs as Configs
# add training directory to runtime system-path
sys.path.append(Configs.path_to_detection_src)

from detect import main, parse_opt


def detect_main(tuple):
    start = time.time()
    opt = parse_opt()
    if tuple[0]:
        opt.source = tuple[0]
    if tuple[1]:
        opt.project = tuple[1]
    opt.cut_event_video = True
    opt.path_to_model_src = Configs.path_to_model_src
    opt.log_path = os.path.join(Configs.path_to_detection_src, 'log')

    opt.weights = Configs.model_weights
    opt.name = tuple[0].split('/')[-3]  # datetime
    print('\n\n[Detect]\n in > %s' % opt.source) 
    print('\n\n[Output]\n in > %s\n\n' % opt.project)

    main(opt)
    return time.time() - start

