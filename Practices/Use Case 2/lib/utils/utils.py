import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path


def time_series_generator(x, y, lookback, steps):
    """
    Method for generating time series data according to the past and future steps

    Parameters:
        x (list): Input array
        y (list): Labels array
        lookback (int): Number of past observations
        steps (int): Number of future steps

    Returns:
        result (list, list): Transformed input array,
    """
    x_transform, y_transform = [], []
    for t in range(len(x)):
        max_step_x = t + lookback
        max_step_y = t + lookback + steps
        if max_step_y <= len(x):
            seq_x, seq_y = x[t:max_step_x], y[max_step_x: max_step_y]
            x_transform.append(seq_x), y_transform.append(seq_y)

    return np.array(x_transform), np.array(y_transform)


def time_series_generator_for_transformers(x, y, lookback, steps):
    """
    Method for generating time series data according to the past and future steps

    Parameters:
        x (list): Input array
        y (list): Labels array
        lookback (int): Number of past observations
        steps (int): Number of future steps

    Returns:
        result (list, list): Transformed input array,
    """
    x_transform, y_transform = [], []
    for t in range(len(x)):
        max_step_x = t + lookback
        max_step_y = t + lookback + steps
        if max_step_y <= len(x):
            seq_x = np.append(x[t:max_step_x], np.zeros((steps, x.shape[1])), axis=0)
            seq_y = y[t: max_step_y, 0]
            x_transform.append(seq_x), y_transform.append(seq_y)

    return np.array(x_transform), np.array(y_transform)


def count_trainable_parameters(model):
    """
    This method calculates the number of parameters of the model.

    Parameters:
        model (torch.nn.Module): Model

    Returns:
        result (int, int): Total number of parameters and number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_experiment_description(logger, FLAGS):
    """
    This method displays the description of the current simulation

    Parameters:
        logger (Logger): logger
        FLAGS (ArgumentParser): argument parser

    Returns:
        Prints the specified simulation parameters
    """
    logger.log_string("-----------------------------------")
    logger.log_string(" - - -  EXPERIMENT PARAMETERS - - - ")
    logger.log_string("-----------------------------------")
    logger.log_string(" - RANDOM_SEED: " + str(FLAGS.random_seed))
    logger.log_string(" - BATCH_SIZE: " + str(FLAGS.batch_size))
    logger.log_string(" - MAX_EPOCH: " + str(FLAGS.max_epoch))
    logger.log_string(" - BASE_LEARNING_RATE: " + str(FLAGS.learning_rate))
    logger.log_string(" - GPU_INDEX: " + str(FLAGS.gpu))
    logger.log_string(" - NUM_CLASSES: " + str(FLAGS.num_classes))
    logger.log_string(" - EARLY_STOPPING: " + str(FLAGS.early_stopping))
    logger.log_string(" - EARLY_STOPPING_PATIENCE: " + str(FLAGS.early_stopping_patience))
    logger.log_string(" - ENERGY_CONSUMPTION: " + str(FLAGS.measure_energy_consumption))
    logger.log_string(" - MODEL: " + str(FLAGS.model))
    logger.log_string(" - DATASET: " + str(FLAGS.dataset_path))
    logger.log_string(" - LOOKBACK: " + str(FLAGS.lookback))
    logger.log_string(" - FUTURE STEPS: " + str(FLAGS.steps))
    logger.log_string(" - SPLIT METHOD: " + str(FLAGS.split_by))
    logger.log_string(" - TRAIN RATIO (how many samples/APs): " + str(FLAGS.train_ratio))
    logger.log_string(" - TEST RATIO (how many samples/APs): " + str(FLAGS.test_ratio))
    logger.log_string("-----------------------------------")


def set_file_name(file_name_str, timestamp_str, file_type_str, FLAGS):
    """
    This method generates a file name for different types of files, indicating relevant information about the experiment

    Parameters:
        file_name_str (String): name of the type of file (e.g., log_train, figure, log_tracker)
        timestamp_str (String): current timestamp
        file_type_str (String): type of file (e.g., txt, csv, jpg)
        FLAGS (ArgumentParser): argument parser

    Returns:
        A string with the file name to be used
    """
    if FLAGS.truncate_steps == 0:
        file_name = file_name_str + '_' + timestamp_str + \
            '_GPU_' + str(FLAGS.gpu) + '_MODEL_' + str(FLAGS.model) + \
            '_DATASET' + str(FLAGS.dataset) + '_EPOCH_' + str(FLAGS.max_epoch) + \
            '_BATCHSIZE_' + str(FLAGS.batch_size) + '_LR_' + str(FLAGS.learning_rate) + \
            '_LOOKBACK_' + str(FLAGS.lookback) + '_STEPS_' + str(FLAGS.steps) + '.' + file_type_str
    elif FLAGS.truncate_steps > 0:
        file_name = file_name_str + '_' + timestamp_str + \
            '_GPU_' + str(FLAGS.gpu) + '_MODEL_' + str(FLAGS.model) + \
            '_DATASET' + str(FLAGS.dataset) + '_EPOCH_' + str(FLAGS.max_epoch) + \
            '_BATCHSIZE_' + str(FLAGS.batch_size) + '_LR_' + str(FLAGS.learning_rate) + \
            '_LOOKBACK_' + str(FLAGS.lookback) + '_STEPS_' + str(FLAGS.truncate_steps) + '.' + file_type_str

    return file_name


def define_date_range(start_date, end_date, delta_hours):
    while start_date < end_date:
        yield start_date
        start_date += timedelta(hours=delta_hours)


def append_statistic_to_file(file_name, statistics, iteration):
    try:
        with open(file_name, 'rb') as handle:
            old_content = pickle.load(handle)
    except (OSError, IOError) as e:
        old_content = {}
    old_content[iteration] = statistics
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    with open(file_name, 'wb') as handle:
        pickle.dump(old_content, handle, protocol=pickle.HIGHEST_PROTOCOL)

