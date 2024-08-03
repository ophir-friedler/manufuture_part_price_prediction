import logging
import json
from pathlib import Path

import pandas as pd
from keras.models import load_model

AVERAGE_TOLERANCE_01_BUCKETED_EXPONENTIAL_BUCKETS = 10
AVERAGE_TOLERANCE_001_BUCKETED_EXPONENTIAL_BUCKETS = 10
AVERAGE_TOLERANCE_0001_BUCKETED_EXPONENTIAL_BUCKETS = 10
LABEL_FEATURE = 'part_price_label'


class ModelServing:
    """
    This class is responsible for serving a model's predictions.
    It is capable of loading a model by name, printing some details, and serving a prediction on various input types.
    """

    def __init__(self):
        self.model = None
        self.model_inputs = None
        self.model_info_df = None
        self.list_of_relu_layer_widths = None
        self.categorical_features_dict = None
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.target_column_name = None
        self.last_neuron_activation = None
        self.loss_function = None
        self.model_inputs = None
        self.model_name = None
        logging.info('Model serving. Empty')

    def pretty_categorical_features_dict(self):
        # create a string with a new line for every key in the dict
        return '\n'.join([f'{key}: {value}' for key, value in self.categorical_features_dict.items()])

    def validate_model_info(self):
        if self.model_name is None:
            raise ValueError('Model name is not set.')
        if self.model_inputs is None:
            raise ValueError('Model inputs and label columns are not set.')

    # Change this according to the environment
    @staticmethod
    def get_models_dir_path_str():
        project_dir = Path(__file__).resolve().parents[2]
        ret_val = str(project_dir / 'models')
        # ret_val = None
        # with open('./mfmatch_api/manu_python/config.json', 'r') as file:
        #     _config = json.load(file)
        #     ret_val = _config['STATIC_DATA_DIR_PATH']
        return ret_val

    @classmethod
    def load_model_by_name(cls, model_name):
        models_dir = cls.get_models_dir_path_str()
        load_path = models_dir + '/' + model_name
        model_info_df = pd.read_parquet(load_path + '/model_info.parquet')
        model_serving = cls()
        model_serving.list_of_relu_layer_widths = model_info_df['list_of_relu_layer_widths'][0]
        model_serving.categorical_features_dict = model_info_df['categorical_features_dict'][0]
        model_serving.epochs = model_info_df['epochs'][0]
        model_serving.batch_size = model_info_df['batch_size'][0]
        model_serving.learning_rate = model_info_df['learning_rate'][0]
        model_serving.target_column_name = model_info_df['target_column_name'][0]
        # model_serving.model_inputs = calculate_model_inputs(model_serving.categorical_features_dict)
        model_serving.model_inputs = list(model_info_df['model_inputs'][0])
        model_serving.model_name = model_info_df['model_name'][0]
        model_serving.model_info_df = model_info_df
        model_serving.model = load_model(load_path + '/' + model_serving.model_name + '.keras')
        return model_serving

    def prepare_data_from_part_features_dict(self, part_features_dict):
        row = self.dict_to_df_row(part_features_dict)
        expanded_row_df = expand_context_target_data(context_target_data_df=row,
                                                     categorical_features=list(self.categorical_features_dict.keys()))
        return self.prepare_expanded_data(expanded_row_df)

    def predict_on_part_measures(self, part_measures_json):
        # item_data_dict = json.loads(part_measures_json)
        measures_list = part_measures_json['measures']
        return self.predict_part_price(self.translate_from_measures_to_features(measures_list))

    def translate_from_measures_to_features(self, measures_list):
        part_features_dict = {}
        tolerance_01 = 0
        tolerance_001 = 0
        tolerance_0001 = 0
        for measure in measures_list:
            upper_tolerance = measure['upper_tolerance']
            lower_tolerance = measure['lower_tolerance']
            tolerance = calculate_tolerance(upper_tolerance, lower_tolerance)
            tolerance_01 += 1 if is_in_tolerance_01(tolerance) else 0
            tolerance_001 += 1 if is_in_tolerance_001(tolerance) else 0
            tolerance_0001 += 1 if is_in_tolerance_0001(tolerance) else 0

        for key, value in self.categorical_features_dict.items():
            if key == 'average_tolerance_01_bucketed':
                part_features_dict[key] = bin_feature(tolerance_01, exponential_bins(AVERAGE_TOLERANCE_01_BUCKETED_EXPONENTIAL_BUCKETS))
            elif key == 'average_tolerance_001_bucketed':
                part_features_dict[key] = bin_feature(tolerance_001, exponential_bins(AVERAGE_TOLERANCE_001_BUCKETED_EXPONENTIAL_BUCKETS))
            elif key == 'average_tolerance_0001_bucketed':
                part_features_dict[key] = bin_feature(tolerance_0001, exponential_bins(AVERAGE_TOLERANCE_0001_BUCKETED_EXPONENTIAL_BUCKETS))
        return part_features_dict

    # Predict price for a single part based on its features
    def predict_part_price(self, part_features_dict):
        logging.info("[NEW] Predicting part price for: " + str(part_features_dict))
        row = self.dict_to_df_row(part_features_dict)
        print('row is: ' + str(row.to_dict()))
        prepared_row = self.prepare_data_from_part_features_dict(part_features_dict)
        print('prepared_row is: ' + str(prepared_row.to_dict()))
        model_input = prepared_row.drop(columns=[LABEL_FEATURE])
        return self.model.predict(model_input, verbose=0), model_input

    def predict_on_prepared_data(self, prepared_data):
        return self.model.predict(prepared_data.drop(columns=[LABEL_FEATURE]))

    def prepare_expanded_data(self, expanded_data):
        # TODO: instead of fill_value=False, create a 'dont know' feature value for each categorical feature
        only_training_columns_df = expanded_data.reindex(columns=self.model_inputs + [LABEL_FEATURE],
                                                         fill_value=False)
        only_training_columns_df.loc[:, self.model_inputs] = only_training_columns_df.loc[:, self.model_inputs].astype(
            'bool')
        # only_training_columns_df.loc[:, self.model_inputs] = only_training_columns_df.loc[:, self.model_inputs].astype(int)
        return only_training_columns_df

    def show_model_details(self):
        print("\nModel features: \n" + str(list(self.categorical_features_dict.keys())))
        print("\n\nModel info: \n " + str(self.model_info_df.to_dict(orient='records')))

    @staticmethod
    def dict_to_df_row(dict_):
        return pd.DataFrame({k: [v] for k, v in dict_.items()})

    @staticmethod
    def string_to_hex(string):
        # Initialize sum to 0
        total_sum = 0

        # Iterate over each character in the string
        for char in string:
            # Convert the character to its binary representation
            binary_value = bin(ord(char))[2:]

            # Convert binary value to integer and add to the total sum
            total_sum += int(binary_value, 2)

        # Convert total sum to hexadecimal and return
        return hex(total_sum)


def expand_context_target_data(context_target_data_df, categorical_features):
    for cat_feature in categorical_features:
        df_expanded = pd.get_dummies(context_target_data_df[cat_feature])
        df_expanded.columns = [f"{cat_feature}_{col}" for col in df_expanded.columns]
        context_target_data_df = pd.concat([context_target_data_df, df_expanded], axis=1)
    return context_target_data_df


def calculate_tolerance(tolerance_upper_deviation, tolerance_lower_deviation):
    return tolerance_upper_deviation - tolerance_lower_deviation


def is_in_tolerance_01(val):
    return val is not None and val >= 0.1


def is_in_tolerance_001(val):
    return val is not None and 0.01 <= val < 0.1


def is_in_tolerance_0001(val):
    return val is not None and val < 0.01


def bin_feature(feature_value, bins_arr):
    bins_arr.sort()
    if feature_value < bins_arr[0]:
        return "<" + str(bins_arr[0])
    for idx, bin_upper_bound in enumerate(bins_arr):
        if feature_value < bin_upper_bound:
            return "[" + str(bins_arr[idx - 1]) + "-" + str(bins_arr[idx]) + ")"
    if feature_value >= bins_arr[-1]:
        return ">=" + str(bins_arr[-1])
    logging.warning("Error: could not bin feature value: " + str(feature_value) + " with bins: " + str(bins_arr))
    # Throw an exception if we got here - we should never get here
    raise Exception("Error: could not bin feature value: " + str(feature_value) + " with bins: " + str(bins_arr))


def exponential_bins(exp_range):
    return [0] + [2 ** i for i in range(0, exp_range)]
