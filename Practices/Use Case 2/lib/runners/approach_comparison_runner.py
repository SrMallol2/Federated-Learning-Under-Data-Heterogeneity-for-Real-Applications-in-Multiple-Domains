import os
import os.path as osp
import pathlib
import pickle
import sched
import subprocess
import sys
import time
from io import StringIO
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd

MAXIMUM_CONCURRENT_JOBS = 2
PROCESS_LIST = []
GPU_CHECK_INTERVAL = 5 * 60
GPU_AVAILABLE_THRESHOLD = 10000
DATASET_PATH = ('/home/dsalami/projects/FedGen/data/FlagsRegression/ApproachComparison/lookback_{lookback}/steps_{'
                'step}')
RESULT_PATH = ('/home/dsalami/projects/FedGen/data/FlagsRegression/ApproachComparison/lookback_{lookback}/steps_{'
                'step}/{approach}/{model}/rep_{rep}')
PYTHON_PATH = '/home/dsalami/.virtualenvs/load prediction/bin/python'
DATASET_GENERATOR_FILE_PATH = '/home/dsalami/projects/FedGen/data/FlagsRegression/generate_niid_dirichlet.py'
TRAIN_FILE_PATH = '/home/dsalami/projects/FedGen/main.py'

DATASET_GENERATOR_TEMPLATE = '\'{python_path}\' \'{dataset_generator_file_path}\'' \
                             ' --destination_path={destination_path} --train_ratio=100' \
                             ' --random_seed=1111 --steps={step} --lookback={lookback} --test_ratio=16'

TRAIN_TEMPLATE = '\'{python_path}\' \'{train_file_path}\' --model={model} --algorithm={approach}' \
                 ' --dataset_path={dataset_path} --problem_type=regression' \
                 ' --steps={steps} --lookback={lookback} --specified_mode=False' \
                 ' --dataset=FlagsRegression-user20-alpha0.0001-ratio1 --num_glob_iters=100' \
                 ' --local_epochs=50 --num_users=20 --device={device} --result_path={result_path}'

HYPER_PARAMETERS = {
    'model': ['lstm', 'cnn'],
    'approach': ['Isolated', 'Centralized', 'FedAvg', 'FedGen'],
    'repetition': [10, 1, 1, 1],
    'lookback': [60],
    'step': [1, 5, 15, 30],
}

DONE_ARRAY = None
AVAILABLE_GPU_QUERY = 'nvidia-smi --query-gpu=index,memory.free --format=csv'

LOG_DIR = '../logs/approach_comparison/'
if not osp.exists(LOG_DIR):
    print('Creating the model checkpoint directory at {}'.format(LOG_DIR))
    makedirs(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log.txt'), 'w')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
    sys.stdout.flush()


def start_next_job_on_gpu(gpu_id):
    global DONE_ARRAY, TRAIN_TEMPLATE, DATASET_GENERATOR_TEMPLATE, DATASET_PATH
    for LOOKBACK_KEY, LOOKBACK in enumerate(HYPER_PARAMETERS['lookback']):
        for STEP_KEY, STEP in enumerate(HYPER_PARAMETERS['step']):
            for MODEL_KEY, MODEL in enumerate(HYPER_PARAMETERS['model']):
                for APPROACH_KEY, APPROACH in enumerate(HYPER_PARAMETERS['approach']):
                    for REPETITION in range(HYPER_PARAMETERS['repetition'][APPROACH_KEY]):
                        RANDOM_SEED = int(
                            '{}{}{}{}{}'.format(LOOKBACK_KEY, STEP_KEY, APPROACH_KEY, MODEL_KEY, REPETITION))
                        if DONE_ARRAY is not None and len(
                                DONE_ARRAY[DONE_ARRAY == RANDOM_SEED]) > 0:
                            continue
                        if DONE_ARRAY is None:
                            DONE_ARRAY = np.array([RANDOM_SEED])
                        else:
                            DONE_ARRAY = np.vstack(
                                (DONE_ARRAY,
                                 np.array([RANDOM_SEED])))
                        log_string(
                            'Starting: lookback={}, step={}, approach={}, model={}, repetition={}, on GPU={}'.format(
                                LOOKBACK,
                                STEP,
                                APPROACH,
                                MODEL,
                                REPETITION,
                                gpu_id))

                        destination_path = DATASET_PATH.format(
                            lookback=LOOKBACK,
                            step=STEP
                        )
                        result_path = RESULT_PATH.format(
                            lookback=LOOKBACK,
                            step=STEP,
                            approach=APPROACH,
                            model=MODEL,
                            rep=REPETITION
                        )
                        dataset_generator_command = DATASET_GENERATOR_TEMPLATE.format(
                            python_path=PYTHON_PATH,
                            dataset_generator_file_path=DATASET_GENERATOR_FILE_PATH,
                            destination_path=destination_path,
                            lookback=LOOKBACK,
                            step=STEP
                        )
                        train_command = TRAIN_TEMPLATE.format(
                            python_path=PYTHON_PATH,
                            train_file_path=TRAIN_FILE_PATH,
                            model=MODEL,
                            approach=APPROACH,
                            dataset_path=destination_path,
                            steps=STEP,
                            lookback=LOOKBACK,
                            device=f'cuda:{gpu_id}',
                            result_path=result_path,
                        )
                        if os.path.exists(result_path):
                            log_string('The model already is trained! Skipped!')
                            continue
                        if os.path.exists(destination_path):
                            command = train_command
                        else:
                            command = f'{dataset_generator_command}; {train_command}'
                        PROCESS_LIST.append(subprocess.Popen(command, shell=True))
                        log_string(command)
                        return True
    return False


def check_available_gpu(scheduler):
    global AVAILABLE_GPU_QUERY, GPU_CHECK_INTERVAL, GPU_AVAILABLE_THRESHOLD, PROCESS_LIST, MAXIMUM_CONCURRENT_JOBS
    log_string('Checking concurrent jobs criterion.')
    current_running_jobs = 0
    finished_processes_indices = []
    for p_index, process in enumerate(PROCESS_LIST):
        poll = process.poll()
        if poll is None:
            current_running_jobs += 1
        else:
            finished_processes_indices.append(p_index)
    for index in sorted(finished_processes_indices, reverse=True):
        del PROCESS_LIST[index]
    if current_running_jobs >= MAXIMUM_CONCURRENT_JOBS:
        log_string('There are {}/{} running jobs. We should wait for one to finish first!'.format(
            MAXIMUM_CONCURRENT_JOBS, MAXIMUM_CONCURRENT_JOBS
        ))
        scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))
        return
    else:
        log_string('There are {}/{} running jobs. Let\'s lunch a new one!!'.format(
            current_running_jobs, MAXIMUM_CONCURRENT_JOBS
        ))
    log_string('GPU availability check...')
    result = subprocess.run(AVAILABLE_GPU_QUERY, stdout=subprocess.PIPE, shell=True)
    std_out_array = StringIO(result.stdout.decode('utf-8'))
    gpu_info = pd.read_csv(std_out_array)
    gpu_info.columns = ['gpu_id', 'free_memory']
    gpu_info['gpu_id'] = gpu_info['gpu_id'].astype(int)
    gpu_info['free_memory'] = gpu_info['free_memory'].str.extract(r'(\d+)', expand=False).astype(float)
    available_gpu = gpu_info[gpu_info['free_memory'] >= GPU_AVAILABLE_THRESHOLD]
    restart_scheduler = False
    if available_gpu.empty:
        log_string('There is no GPU available at the moment.')
        restart_scheduler = True
    else:
        log_string('{} GPU(s) is(are) available for the next job.'.format(len(available_gpu.index)))
        if not start_next_job_on_gpu(available_gpu.loc[available_gpu.index[0], 'gpu_id']):
            log_string('There is no job to run! Terminating the program!')
        else:
            restart_scheduler = True
    if restart_scheduler:
        scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))


scheduler = sched.scheduler(time.time, time.sleep)
scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))
check_available_gpu(scheduler)
scheduler.run()
