import itertools
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split


class PartPricePredictor:
    _label_column = 'Rate (EURO) mean_netsuite_496' # 'Rate mean_netsuite'  #
    _categorical_features = None
    _input_table_name = 'part_price_training_table_496' # 'part_price_training_table'  #
    _training_table_name = _input_table_name + '_training'
    _is_model_trained = False
    _last_neuron_function = 'linear'
    _model = None

    _all_part_features = [
        # ## Part features
        # 'coc'
        # , 'first_material_categorization_level_1_set'
        # , 'first_material_categorization_level_2_set'
        # , 'first_material_categorization_level_3_set'
        # , 'average_number_of_nominal_sizes'
        # , 'average_number_of_nominal_sizes_bucketed'
        # , 'average_tolerance_01'
        # , 'average_tolerance_01_bucketed'
        # , 'average_tolerance_001_bucketed'
        # , 'average_tolerance_0001_bucketed'
        # ,
        # 'average_tolerance'
        'max_enclosing_cuboid_volume_bucketed'
    ]

    # _selected_singles = [
    #     # ## Part features
    #     # 'coc'
    #     # , 'first_material_categorization_level_1_set'
    #     # , 'first_material_categorization_level_2_set'
    #     # , 'first_material_categorization_level_3_set'
    #     # , 'average_number_of_nominal_sizes'
    #     # , 'average_number_of_nominal_sizes_bucketed'
    #     # , 'average_tolerance_01'
    #     # , 'average_tolerance_01_bucketed'
    #     # , 'average_tolerance_001_bucketed'
    #     # , 'average_tolerance_0001_bucketed'
    #     # ,
    #     'max_enclosing_cuboid_volume_bucketed'
    # ]

    _selected_doubles = [
        # ['post_id_manuf', 'plan']
        # , ['post_id_manuf', 'req_sheet_metal_inserts']
        # , ['post_id_manuf', 'req_sheet_metal']
        # , ['post_id_manuf', 'one_manufacturer']
        # , ['post_id_manuf', 'num_distinct_parts_binned']
        # , ['post_id_manuf', 'total_quantity_of_parts_binned']
        # , ['sheet_metal_weldings', 'sheet_metal_punching']
    ]

    def __init__(self, all_part_features=None):
        # Singles
        self._list_of_relu_layer_widths = None
        self._x_train_two_rows = None
        self._model_input_columns = None
        if all_part_features is None:
            self._all_used_features = self._selected_singles = self._all_part_features
        else:
            self._all_used_features = self._selected_singles = self._all_part_features = all_part_features
        # Feature selection
        self._all_training_features = self._selected_singles + self.get_selected_double_feature_names()
        # Categorical
        self._categorical_features = self._all_training_features

    def __str__(self):
        ret_str = "_input_table_name: " + self._input_table_name + '\n' + " _label_column: " + self._label_column \
                  + " _training_table_name: " + self._training_table_name \
                  + " _model: " + str(self._model)
        return ret_str

    def save_model(self, model_output_path):
        model_name = 'model__' + str(self._list_of_relu_layer_widths) + '__' + 'T__' + self._training_table_name
        model_save_path = model_output_path + "/" + model_name + '.h5'
        self._model.save(model_save_path)
        # save the fit_predict_columns to a file and maintain the order of the columns
        self._x_train_two_rows.to_parquet(model_output_path + "/" + model_name + '_x_train_two_rows.parquet')
        with open(model_output_path + "/" + model_name + ' _model_input_columns.json', 'w') as file:
            json.dump(self._model_input_columns, file)

        logging.warning("Saved model to: " + model_save_path)

    # def load_model(self, model_path):
    #     self._model = tf.keras.models.load_model(model_path)
    #     model_name = model_path.split('/')[-1].split('.')[0]
    #     list_of_relu_layer_widths_str = model_path.split('__')[1]
    #     # convert string to list of integers
    #     self._list_of_relu_layer_widths = [int(layer_width) for layer_width in list_of_relu_layer_widths_str.split(',')]
    #     self._is_model_trained = True
    #     self._x_train_two_rows = pd.read_parquet(STATIC_DATA_DIR_PATH + model_name + '_x_train_two_rows.parquet')
    #     with open(STATIC_DATA_DIR_PATH + '_model_input_columns.json', 'r') as file:
    #         self._model_input_columns = json.load(file)

    def _validate_configuration(self):
        all_features_set = set(self._selected_singles).union(
            set([single for double in self._selected_doubles for single in double]))
        sym_dif = set(self._all_used_features).symmetric_difference(all_features_set)
        if len(sym_dif) > 0:
            logging.error("inconsistency in the following features : " + str(sym_dif))
            return False
        return True

    def build_model(self, all_tables_df, list_of_relu_layer_widths, epochs, batch_size, evaluate=False, verbose=False):
        self._list_of_relu_layer_widths = list_of_relu_layer_widths
        if self._validate_configuration():
            all_tables_df[self._training_table_name] = self.prepare_for_fit_predict(
                all_tables_df[self._input_table_name], verbose=verbose)
            if verbose:
                print("Training data table name: " + self._training_table_name)

            X_train = all_tables_df[self._training_table_name].drop(columns=[self._label_column])
            y_train = all_tables_df[self._training_table_name][self._label_column]
            X_test = None
            y_test = None
            if evaluate:
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
                self._x_train_two_rows = X_train.head(2)
            self._x_train_two_rows = X_train.head(2)
            self._model_input_columns = {column_name: column_index for column_index, column_name in enumerate(X_train.columns)}

            input_dim = len(X_train.columns)

            # validate model instance data
            if list_of_relu_layer_widths is None or len(list_of_relu_layer_widths) == 0 or input_dim == 0:
                logging.error("Model instance shape (input_dim: " + str(input_dim) + ", layers: "
                              + str(list_of_relu_layer_widths) + "), problematic")
                return None

            self._model = self.build_model_instance(input_dim, list_of_relu_layer_widths)

            # Compile the model
            self._model.compile(optimizer='adam',
                                loss='mean_squared_error')
            if verbose:
                print("Model: " + str(self._model))
            self._model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
            self._is_model_trained = True
            if evaluate:
                self._model.evaluate(X_test, y_test, verbose=2)
                # Make predictions on evaluation data
                predictions = self._model.predict(X_test)
                # Calculate the ratio between actual values and predictions
                ratio = y_test / predictions.flatten()
                # for each sample, take the max of the two ratios
                ratio = ratio.apply(lambda x: max(x, 1 / x))
                # cap ratio values at 10
                ratio = ratio.apply(lambda x: 10 if x > 10 else x)

                # Visualize the distribution of the ratio
                plt.figure(figsize=(10, 6))
                sns.histplot(ratio, bins=100, kde=True)
                plt.title('Distribution of Ratio (Actual price / Predicted price)')
                plt.xlabel('Ratio')
                plt.ylabel('Frequency')
                plt.show()

            return self._model

    def get_selected_double_feature_names(self):
        return [get_double_feature_name(column_a, column_b) for [column_a, column_b] in self._selected_doubles]

    def prepare_doubles(self, raw_data):
        for [column_a, column_b] in self._selected_doubles:
            raw_data[get_double_feature_name(column_a, column_b)] = raw_data.apply(
                lambda row: get_double_feature_value_new(row, column_a, column_b), axis='columns')
        return raw_data

    def do_all_features_exist(self, columns):
        cols_difference_set = set(self._all_used_features).difference(set(columns))
        if len(cols_difference_set) > 0:
            logging.error("missing columns: " + str(cols_difference_set))
            return False
        return True

    def model_predict(self, predict_input):
        logging.info("Predicting with model: " + str(self._model))
        return self._model.predict(predict_input, verbose=0)

    def build_model_instance(self, input_dim, list_of_relu_layer_widths):
        model = Sequential()
        model.add(Dense(list_of_relu_layer_widths[0], activation='relu', input_dim=input_dim))
        for layer_width in list_of_relu_layer_widths[1:]:
            model.add(Dense(layer_width, activation='relu'))
        model.add(Dense(1, activation=self._last_neuron_function))
        return model

    # Predict price for a single part based on its features
    def predict_part_price(self, part_features_dict):
        row = pd.DataFrame.from_dict(part_features_dict)
        prepared_row = self.prepare_for_fit_predict(row)
        return self.model_predict(prepared_row)

    def price_predictions_for_all_feature_combinations(self, all_tables_df, csv_filename=None):
        print("Started price_predictions_for_all_feature_combinations")
        # collect all feature values for all features, and then create a cartesian product of all feature values
        all_values = {feature: all_tables_df[self._input_table_name][feature].unique() for feature in self._all_part_features}

        combinations = itertools.product(*[all_values[feature] for feature in self._all_part_features])

        ret_df = pd.DataFrame(combinations, columns=self._all_part_features)

        # Predict prices for all feature combinations
        ret_df['pred_price'] = ret_df.apply(lambda row:
                                            self.predict_part_price({k: [v] for k, v in row.to_dict().items()}), axis=1)

        # for element in combinations:
        #     part_features_dict = {k: [v] for k, v in zip(self._all_part_features, element)}
        #     predict_rows = self.predict_part_price(part_features_dict)
        #     predict_rows = pd.DataFrame(predict_rows, columns=['pred_price'])
        #     predict_rows = pd.concat([pd.DataFrame(part_features_dict), predict_rows], axis=1)
        #     ret_df = pd.concat([ret_df, predict_rows])
        if csv_filename is not None:
            ret_df.to_csv(csv_filename, index=False)
        return ret_df
        # display a graph that shows the distribution of prices as a function of tolerance_bucketed
        # sns.scatterplot(data=ret_df, x='average_tolerance_01_bucketed', y='pred_price')

    # Complete columns required for predicting from model (if missing then should be 0 in a 1-hot encoding)
    def complete_columns_with_negatives(self, prepared_data):
        if self._x_train_two_rows is not None:
            missing_columns = list(set(self._x_train_two_rows.columns).difference(set(prepared_data.columns)))
            missing_columns_1 = list(set(self._model_input_columns.keys()).difference(set(prepared_data.columns)))
            # check if the missing columns are the same as the ones in the model_input_columns
            if len(missing_columns) != len(missing_columns_1) \
                    or len(set(missing_columns).symmetric_difference(set(missing_columns_1))) > 0:
                logging.error("inconsistency in the missing columns : " + str(missing_columns) + " vs " + str(missing_columns_1))
            else:
                logging.error("columns from two rows : " + str(self._x_train_two_rows.columns))
                logging.error("columns from model_input_columns : " + str(self._model_input_columns.keys()))

            prepared_data = pd.concat([prepared_data, pd.DataFrame(index=prepared_data.index, columns=missing_columns)],
                                      axis=1)
            prepared_data = prepared_data.fillna(0)

            # Reorder (and filter) columns to the order at train time
            prepared_data = prepared_data[self._x_train_two_rows.columns]
        else:
            logging.error("Trying to complete columns before model is trained")
        return prepared_data

    # Feature validation
    # Only necessary features
    # One hot encoding
    def prepare_for_fit_predict(self, input_table_rows_df, verbose=False):
        # Feature validation
        if self.do_all_features_exist(input_table_rows_df.columns):
            ret_df = input_table_rows_df.copy()
            ret_df = self.prepare_doubles(ret_df)
            if verbose:
                print("Before one hot encoding: ")
                print(len(list(ret_df.columns)))
                print(list(ret_df.columns))
                print("All training features: ")
                print(self._all_training_features)
                print("Label column")
                print(self._label_column)
                print("Categorical: ")
                print(self._categorical_features)

            # Only necessary features
            ret_df = ret_df[ret_df.columns.intersection(self._all_training_features + [self._label_column])]
            if verbose:
                print("ret_df after intersection")
                print(len(list(ret_df.columns)))
                print(list(ret_df.columns))

            # One hot encoding
            for categorical_feature in self._categorical_features:
                if categorical_feature in ret_df.columns:
                    ret_df = pd.concat([ret_df, pd.get_dummies(ret_df[categorical_feature],
                                                               prefix=categorical_feature)],
                                       axis=1).drop(columns=[categorical_feature])
            ret_df = self.complete_columns_with_negatives(prepared_data=ret_df)
            return ret_df
        logging.error("Features missing in data passed to prepare_for_fit_predict")
        return pd.DataFrame()

    @property
    def model(self):
        return self._model


def get_double_feature_name(column_a, column_b):
    return column_a + "__" + column_b


def get_double_feature_value(df, column_a, column_b):
    return df[column_a].astype("string") + "_" + \
           df[column_b].astype("string")


def get_double_feature_value_new(ser, column_a, column_b):
    return str(ser[column_a]) + "_" + str(ser[column_b])
