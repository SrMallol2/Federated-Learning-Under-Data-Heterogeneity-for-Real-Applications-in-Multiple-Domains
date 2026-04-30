import pickle


def FlagsRegressionInversePreprocessing(base_path, split, prediction, axis=None):
    with open('{}/{}_scaler.pkl'.format(base_path, split), 'rb') as handle:
        scaler = pickle.load(handle)
    unscaled_data = scaler.inverse_transform(prediction)
    if axis is not None:
        unscaled_data = unscaled_data[:, 0]
    unscaled_data = 10 ** unscaled_data
    return unscaled_data


INVERSE_PREPROCESSING = {
    'flags-regression': FlagsRegressionInversePreprocessing
}
