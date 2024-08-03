import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras import Sequential, optimizers
from keras.src.layers import Dense

from src.data import dal
from src.features.build_features import transform_to_comma_separated_str_set, get_first_material_category_level_1_set
from src.models.part_price_model_serving import ModelServing, LABEL_FEATURE


def model_exists(model_name):
    project_dir = Path(__file__).resolve().parents[2]
    return (project_dir / 'models' / model_name).exists()


class ModelHandler(ModelServing):
    """
    This class is responsible for handling a model.
    It is responsible for training the model, predicting on the data and saving the model.
    It is also capable of loading a model by name, and saving a model.
    """

    def __init__(self, list_of_relu_layer_widths=None,
                 categorical_features_dict=None,
                 epochs=None,
                 batch_size=None,
                 learning_rate=None,
                 target_column_name=None,
                 last_neuron_activation=None,
                 loss_function=None
                 ):
        super().__init__()
        self.model = None
        self.model_inputs = None
        self.model_info_df = None

        self.list_of_relu_layer_widths = list_of_relu_layer_widths
        self.categorical_features_dict = categorical_features_dict
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_column_name = target_column_name
        self.last_neuron_activation = last_neuron_activation
        self.loss_function = loss_function
        logging.info(f"Using following target column name: {self.target_column_name}")
        self.model_inputs = calculate_model_inputs(self.categorical_features_dict)
        self.model_name = self.get_model_name()
        logging.info('Model handler created. model name: ' + self.model_name
                     + ', categorical_features_dict: '
                     + str(self.categorical_features_dict) + ', target_column_name: ' + str(self.target_column_name))

    def get_model_name(self):
        # create a hash from the list self.model_inputs
        # and the list self.list_of_relu_layer_widths

        return 'MA_' + '_'.join([str(x) for x in self.list_of_relu_layer_widths]) \
               + '_MIH_' + str(self.string_to_hex(str(self.model_inputs))) \
               + '_TH_' + str(self.string_to_hex(str(self.target_column_name))) \
               + '_E_' + str(self.epochs) \
               + '_BS_' + str(self.batch_size) \
               + '_LR_' + str(self.learning_rate)

    def validate_model_info(self):
        super().validate_model_info()
        if self.model_name is None:
            raise ValueError('Model name is not set.')
        if self.list_of_relu_layer_widths is None:
            raise ValueError('List of relu layer widths is not set.')
        if self.epochs is None:
            raise ValueError('Epochs is not set.')
        if self.batch_size is None:
            raise ValueError('Batch size is not set.')
        if self.learning_rate is None:
            raise ValueError('Learning rate is not set.')
        if self.model_inputs is None:
            raise ValueError('Model inputs and label columns are not set.')

    @classmethod
    def get_trained_model(cls,
                          list_of_relu_layer_widths,
                          epochs=10,
                          batch_size=32,
                          learning_rate=0.01,
                          target_column_name=None,
                          last_neuron_activation=None,
                          loss_function=None,
                          training_table_name=None,
                          categorical_features=None,
                          verbose=False):
        categorical_features_dict = {categorical_feature: dal.get_distinct_values(table_name=training_table_name,
                                                                                  column_name=categorical_feature).iloc[
                                                          :, 0].tolist()
                                     for categorical_feature in categorical_features}
        model_handler = cls(list_of_relu_layer_widths=list_of_relu_layer_widths,
                            categorical_features_dict=categorical_features_dict,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            target_column_name=target_column_name,
                            last_neuron_activation=last_neuron_activation,
                            loss_function=loss_function
                            )
        if list_of_relu_layer_widths is not None:
            model_handler.build_and_train_model(training_table_name=training_table_name, verbose=verbose)
        return model_handler

    # @classmethod
    # def load_model_by_name(cls, model_name):
    #     project_dir = Path(__file__).resolve().parents[2]
    #     load_path = project_dir / 'models' / model_name
    #     model_info_df = pd.read_parquet(load_path / 'model_info.parquet')
    #     model_handler = cls(list_of_relu_layer_widths=model_info_df['list_of_relu_layer_widths'][0],
    #                         categorical_features_dict=model_info_df['categorical_features_dict'][0],
    #                         epochs=model_info_df['epochs'][0],
    #                         batch_size=model_info_df['batch_size'][0],
    #                         learning_rate=model_info_df['learning_rate'][0],
    #                         target_column_name=model_info_df['target_column_name'][0])
    #     model_handler.model_inputs = list(model_info_df['model_inputs'][0])
    #     model_handler.model_name = model_info_df['model_name'][0]
    #     model_handler.model_info_df = model_info_df
    #     model_handler.model = load_model(load_path / (model_handler.model_name + '.keras'))
    #     return model_handler

    @classmethod
    def delete_model_by_name(cls, model_name):
        project_dir = Path(__file__).resolve().parents[2]
        # Check if model exists and if so, delete it
        if model_exists(model_name):
            load_path = project_dir / 'models' / model_name
            shutil.rmtree(load_path)
            logging.info('Model ' + model_name + ' deleted.')
        else:
            logging.warning('Model ' + model_name + ' does not exist.')

    def evaluate_model(self, evaluation_table_name):
        evaluation_df = self.prepare_data_from_training_table(training_table_name=evaluation_table_name)
        X_test = evaluation_df.drop(columns=[LABEL_FEATURE])
        y_test = evaluation_df[LABEL_FEATURE]
        self.model.evaluate(X_test, y_test, verbose=2)
        predictions = self.model.predict(X_test)
        ratio = y_test / predictions.flatten()
        ratio = ratio.apply(lambda x: max(x, 1 / x))
        ratio = ratio.apply(lambda x: 10 if x > 10 else x)
        combined = pd.concat([X_test, y_test], axis=1)
        combined['predicted'] = predictions

        # Visualize the distribution of the predicted prices
        plt.figure(figsize=(10, 6))
        sns.histplot(combined['predicted'], bins=100, kde=True)
        plt.title('Distribution of Predicted Prices')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

        # Visualize the distribution of the ratio
        plt.figure(figsize=(10, 6))
        sns.histplot(ratio, bins=100, kde=True)
        plt.title('Distribution of Ratio (Actual price / Predicted price)')
        plt.xlabel('Ratio')
        plt.ylabel('Frequency')
        plt.show()

    def save_model(self):
        project_dir = Path(__file__).resolve().parents[2]
        save_path = project_dir / 'models' / self.model_name
        save_path.mkdir(parents=True, exist_ok=True)
        model_file_name = self.model_name + '.keras'
        logging.info(f'Saving model name {self.model_name} to ' + str(save_path))
        self.model.save(save_path / model_file_name)
        model_info_df = pd.DataFrame({'model_inputs': [self.model_inputs],
                                      'model_name': [self.model_name],
                                      'list_of_relu_layer_widths': [self.list_of_relu_layer_widths],
                                      'categorical_features_dict': [self.categorical_features_dict],
                                      'epochs': [self.epochs],
                                      'batch_size': [self.batch_size],
                                      'learning_rate': [self.learning_rate],
                                      'target_column_name': [self.target_column_name],
                                      'last_neuron_activation': [self.last_neuron_activation],
                                      'loss_function': [self.loss_function]
                                      }
                                     )
        model_info_df.to_parquet(save_path / 'model_info.parquet')

    def compile_model(self):
        self.validate_model_info()
        input_dim = len(self.model_inputs)
        model = Sequential()
        model.add(Dense(self.list_of_relu_layer_widths[0], activation='relu', input_dim=input_dim))
        for layer_width in self.list_of_relu_layer_widths[1:]:
            model.add(Dense(layer_width, activation='relu'))
        model.add(Dense(1, activation=self.last_neuron_activation))

        # Compile a classification model
        logging.info('Compiling model with learning rate: ' + str(self.learning_rate))
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        # TODO: enable this option: Use SGD optimizer with learning rate 0.01
        # optimizer = optimizers.SGD(learning_rate=0.01)
        # TODO: maybe this? #   'model_structure': ['LSTM', 'GRU']
        model.compile(optimizer=optimizer,
                      loss=self.loss_function)
        self.model = model
        return model

    def build_and_train_model(self,
                              training_table_name=None,
                              verbose=False):
        self.validate_model_info()
        self.compile_model()
        logging.info(f'Starting to train a model on {training_table_name} with num columns: {len(self.model_inputs)}')
        self.train_model(training_table_name, verbose)

    def train_model(self, training_table_name, verbose=False):
        self.validate_model_info()
        if verbose:
            logging.info(f'\n\n Training model based on table: {training_table_name} \n\n')
        ret_val = self.fit_on_prepared_data(
            self.prepare_data_from_training_table(training_table_name=training_table_name,
                                                  verbose=verbose))
        if ret_val == 'no_data':
            logging.warning(f'No data to train on from table: {training_table_name}')
        return ret_val

    def fit_on_prepared_data(self, prepared_data):
        if len(prepared_data) == 0:
            logging.warning('No data to train on. Skipping training.')
            return 'no_data'
        try:
            self.model.fit(prepared_data.drop(columns=[LABEL_FEATURE]),
                           prepared_data[LABEL_FEATURE],
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           verbose=1)
            return 'Success'
        except Exception as e:
            logging.error(f'Error while fitting model: {e}')
            return 'Fail'

    def prepare_data_from_training_table(self, training_table_name, verbose=False):
        features = list(self.categorical_features_dict.keys())
        context_target_data_df = build_context_target_data_direct(training_table_name=training_table_name,
                                                                  categorical_features=features,
                                                                  target_column_name=self.target_column_name,
                                                                  verbose=verbose)
        if verbose:
            report_str = " first line"
            report_str += ", second line"
            logging.info(report_str)
        expanded_training_data_df = expand_context_target_data(context_target_data_df=context_target_data_df,
                                                               categorical_features=features)
        prepared_data = self.prepare_expanded_data(expanded_data=expanded_training_data_df)
        return prepared_data

    # def prepare_data_from_part_features_dict(self, part_features_dict):
    #     row = util_functions.dict_to_df_row(part_features_dict)
    #     expanded_row_df = expand_context_target_data(context_target_data_df=row,
    #                                                  categorical_features=list(self.categorical_features_dict.keys()))
    #     return self.prepare_expanded_data(expanded_row_df)

    # Predict price for a single part based on its features
    # def predict_part_price(self, part_features_dict):
    #     logging.info("[NEW] Predicting part price for: " + str(part_features_dict))
    #     row = util_functions.dict_to_df_row(part_features_dict)
    #     print('row is: ' + str(row.to_dict()))
    #     prepared_row = self.prepare_data_from_part_features_dict(part_features_dict)
    #     print('prepared_row is: ' + str(prepared_row.to_dict()))
    #     model_input = prepared_row.drop(columns=[LABEL_FEATURE])
    #     return self.model.predict(model_input, verbose=0), model_input
    #
    def predict_on_werk_raw_data(self, werk_raw_dict):
        return self.predict_part_price(convert_werk_raw_to_categorical_features_dict(werk_raw_dict))
    #
    # def predict_on_prepared_data(self, prepared_data):
    #     return self.model.predict(prepared_data.drop(columns=[LABEL_FEATURE]))
    #
    # def prepare_expanded_data(self, expanded_data):
    #     # TODO: instead of fill_value=False, create a 'dont know' feature value for each categorical feature
    #     only_training_columns_df = expanded_data.reindex(columns=self.model_inputs + [LABEL_FEATURE],
    #                                                      fill_value=False)
    #     only_training_columns_df.loc[:, self.model_inputs] = only_training_columns_df.loc[:, self.model_inputs].astype(
    #         'bool')
    #     # only_training_columns_df.loc[:, self.model_inputs] = only_training_columns_df.loc[:, self.model_inputs].astype(int)
    #     return only_training_columns_df
    #
    # def show_model_details(self):
    #     print("\nModel features: \n" + str(list(self.categorical_features_dict.keys())))
    #     print("\n\nModel info: \n " + str(self.model_info_df.to_dict(orient='records')))


# def string_to_hex(string):
#     # Initialize sum to 0
#     total_sum = 0
#
#     # Iterate over each character in the string
#     for char in string:
#         # Convert the character to its binary representation
#         binary_value = bin(ord(char))[2:]
#
#         # Convert binary value to integer and add to the total sum
#         total_sum += int(binary_value, 2)
#
#     # Convert total sum to hexadecimal and return
#     return hex(total_sum)


def calculate_model_inputs(categorical_features_dict):
    columns_to_train = []

    logging.info('Using following categorical features: ' + str(categorical_features_dict))
    for feature, values in categorical_features_dict.items():
        for value in values:
            columns_to_train.extend([f"{feature}_{value}"])
    return columns_to_train


def drop_all_models():
    logging.info('Dropping all models')
    project_dir = Path(__file__).resolve().parents[2]
    models_dir = project_dir / 'models'
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            for model_file in model_dir.iterdir():
                if model_file.is_file():
                    model_file.unlink()
            model_dir.rmdir()
    logging.info('All models dropped')


def clean_away_model_by_name(model_name, verbose=False):
    if model_exists(model_name):
        logging.info(f'Deleting model {model_name}')
        ModelHandler.delete_model_by_name(model_name)
    else:
        logging.warning(f'Model {model_name} does not exist. Skipping deletion.') if verbose else None


def forget_model_by_name(model_name, verbose=False):
    if model_exists(model_name):
        logging.info(f'Forgetting model {model_name}')
        clean_away_model_by_name(model_name=model_name, verbose=verbose)
    else:
        logging.warning(f'Model {model_name} does not exist. Skipping deletion.') if verbose else None


def build_context_target_data_direct(training_table_name, categorical_features, target_column_name, verbose):
    # Target
    query = f"""
        SELECT
            post_id,  -- Info column
            {target_column_name} AS {LABEL_FEATURE},
    """

    # Context
    query += ', '.join([f'{feature}' for feature in categorical_features])

    query += f"""
        FROM {training_table_name}
    """
    logging.info("Query: " + query) if verbose else None
    return dal.read_query(query_str=query)


def expand_context_target_data(context_target_data_df, categorical_features):
    for cat_feature in categorical_features:
        df_expanded = pd.get_dummies(context_target_data_df[cat_feature])
        df_expanded.columns = [f"{cat_feature}_{col}" for col in df_expanded.columns]
        context_target_data_df = pd.concat([context_target_data_df, df_expanded], axis=1)
    return context_target_data_df


def convert_werk_raw_to_categorical_features_dict(werk_raw_dict):
    werk_dict = {'name': werk_raw_dict['name']}
    title_block = werk_raw_dict['title_block']
    if title_block is None or title_block['material'] is None or title_block['material']['material_category'] is None:
        werk_dict['material_category_level_1'] = []
        werk_dict['material_category_level_2'] = []
        werk_dict['material_category_level_3'] = []
    else:
        werk_dict['material_category_level_1'] = [title_block['material']['material_category'][0]]
        werk_dict['material_category_level_2'] = [title_block['material']['material_category'][1]]
        werk_dict['material_category_level_3'] = [title_block['material']['material_category'][2]]
    werk_df = pd.DataFrame(werk_dict)
    werk_by_name_df = werk_df.groupby('name').agg(
        material_category_level_1_set=('material_category_level_1', lambda x: transform_to_comma_separated_str_set(x)),
        material_category_level_2_set=('material_category_level_2', lambda x: transform_to_comma_separated_str_set(x)),
        material_category_level_3_set=('material_category_level_3', lambda x: transform_to_comma_separated_str_set(x)))
    ret_ser = pd.Series({'num_werk_results': len(werk_by_name_df),
                         'first_material_category_level_1_set': get_first_material_category_level_1_set(werk_by_name_df)
                         })
    logging.info(f"Converted werk raw to categorical features dict: {ret_ser.to_dict()}")
    return ret_ser.to_dict()
