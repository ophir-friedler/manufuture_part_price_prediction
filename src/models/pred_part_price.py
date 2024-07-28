# import itertools
# import json
# import logging
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import tensorflow as tf
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.models import load_model
# from sklearn.model_selection import train_test_split
#
# from src.utils import util_functions
#
#
# class PartPricePredictor:
#     _label_column = 'unit_price'
#     _categorical_features = None
#     _input_table_name = 'part_price_training_table_646'
#     _training_table_name = _input_table_name + '_training'
#     _is_model_trained = False
#     _last_neuron_activation = 'linear'
#     _model = None
#
#     _all_part_features = [
#         # ## Part features
#         'first_material_category_level_1_set',
#         # 'coc'
#         # , 'first_material_category_level_2_set'
#         # , 'first_material_category_level_3_set'
#         # , 'average_number_of_nominal_sizes'
#         # , 'average_number_of_nominal_sizes_bucketed'
#         # , 'average_tolerance_01'
#         # , 'average_tolerance_01_bucketed'
#         # , 'average_tolerance_001_bucketed'
#         # , 'average_tolerance_0001_bucketed'
#         # ,
#         # 'average_tolerance'
#         'max_enclosing_cuboid_volume_bucketed'
#     ]
#
#     def __init__(self, list_of_relu_layer_widths=None, all_part_features=None, model_input_columns=None,
#                  model_input_columns_and_label=None):
#         self._list_of_relu_layer_widths = list_of_relu_layer_widths
#         self._model_input_columns_and_label = model_input_columns_and_label
#         self.batch_size = None
#         self.epochs = None
#         self._x_train_two_rows = None
#         if model_input_columns is not None:
#             logging.info(f"loaded model input columns. Length: {len(model_input_columns)}")
#             self._model_input_columns = model_input_columns
#         self._model_input_columns = None
#         if all_part_features is None:
#             self._all_used_features = self._all_part_features
#         else:
#             self._all_used_features = self._all_part_features = all_part_features
#         # Feature selection
#         self._all_training_features = self._all_part_features
#         # Categorical
#         self._categorical_features = self._all_training_features
#         logging.info("Categorical features: " + str(self._categorical_features))
#         logging.info("All training features: " + str(self._all_training_features))
#         logging.info("All part features: " + str(self._all_part_features))
#         logging.info("All used features: " + str(self._all_used_features))
#         self.model_name = self.get_model_name()
#
#     def __str__(self):
#         ret_str = "_input_table_name: " + self._input_table_name + '\n' + " _label_column: " + self._label_column \
#                   + " _training_table_name: " + self._training_table_name \
#                   + " _model: " + str(self._model)
#         return ret_str
#
#     def get_model_name(self):
#         return 'model__' + str(util_functions.string_to_hex(str(self._list_of_relu_layer_widths))) + \
#                '__' + str(util_functions.string_to_hex(str(self._all_part_features)))
#
#     def save_model(self, model_output_path):
#         model_name = self.get_model_name()
#         model_save_path = model_output_path + "/" + model_name + '.h5'
#         self._model.save(model_save_path)
#         # save the fit_predict_columns to a file and maintain the order of the columns
#         self._x_train_two_rows.to_parquet(model_output_path + "/" + model_name + '_x_train_two_rows.parquet')
#         with open(model_output_path + "/" + model_name + ' _model_input_columns.json', 'w') as file:
#             json.dump(self._model_input_columns, file)
#
#         logging.warning("Saved model to: " + model_save_path)
#
#     def save_model_new(self):  # model_output_path
#         project_dir = Path(__file__).resolve().parents[2]
#         self.model_name = self.get_model_name()
#         save_path = project_dir / 'models' / self.model_name
#         save_path.mkdir(parents=True, exist_ok=True)
#         model_file_name = self.model_name + '.keras'
#         logging.info(f'Saving model {self.model_name} to ' + str(save_path))
#         self._model.save(save_path / model_file_name)
#         model_info_dict = {'_label_column': [self._label_column],
#                            '_categorical_features': [self._categorical_features],
#                            '_input_table_name': [self._input_table_name],
#                            '_training_table_name': [self._training_table_name],
#                            '_is_model_trained': [self._is_model_trained],
#                            '_last_neuron_activation': [self._last_neuron_activation],
#                            '_list_of_relu_layer_widths': [self._list_of_relu_layer_widths],
#                            '_all_used_features': [self._all_used_features],
#                            '_all_training_features': [self._all_training_features],
#                            '_model_input_columns': [self._model_input_columns],
#                            '_model_input_columns_and_label': [self._model_input_columns_and_label],
#                            '_all_part_features': [self._all_part_features]
#                            }
#         logging.info("model_info_dict: " + str(model_info_dict))
#         model_info_df = pd.DataFrame(model_info_dict)
#         model_info_df.to_parquet(save_path / 'model_info.parquet')
#
#     @classmethod
#     def load_model_by_name(cls, model_name):
#         logging.info("Loading model by name: " + model_name)
#         project_dir = Path(__file__).resolve().parents[2]
#         load_path = project_dir / 'models' / model_name
#         model_info_df = pd.read_parquet(load_path / 'model_info.parquet')
#         print(model_info_df.to_dict(orient='records'))
#         model_handler = cls(list_of_relu_layer_widths=list(model_info_df['list_of_relu_layer_widths'][0]),
#                             all_part_features=list(model_info_df['_all_part_features'][0]),
#                             model_input_columns_and_label=list(model_info_df['_model_input_columns_and_label'][0]))
#         # model_handler.model_inputs = list(model_info_df['model_inputs'][0])
#         model_handler._model = load_model(load_path / (model_handler.model_name + '.keras'))
#         return model_handler
#
#     def _validate_configuration(self):
#         all_features_set = set(self._all_part_features)
#         sym_dif = set(self._all_used_features).symmetric_difference(all_features_set)
#         if len(sym_dif) > 0:
#             logging.error("inconsistency in the following features : " + str(sym_dif))
#             return False
#         return True
#
#     def build_model(self, all_tables_df, list_of_relu_layer_widths, epochs, batch_size, all_part_features,
#                     evaluate=False, verbose=False):
#         self._list_of_relu_layer_widths = list_of_relu_layer_widths
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self._all_part_features = all_part_features
#         self._model_input_columns_and_label = None
#         if self._validate_configuration():
#             logging.info("training table name: " + self._training_table_name) if verbose else None
#             logging.info("input table name: " + self._input_table_name) if verbose else None
#             logging.info(
#                 "input table columns before prepare: " + str(list(all_tables_df[self._input_table_name].columns)))
#             logging.info(self._all_used_features)
#
#             all_tables_df[self._training_table_name] = self.prepare_for_fit_predict(
#                 all_tables_df[self._input_table_name], verbose=verbose)
#             self._model_input_columns_and_label = list(all_tables_df[self._training_table_name].columns)
#             if verbose:
#                 print("Training data table name: " + self._training_table_name)
#
#             print("input table head as dict")
#             print(all_tables_df[self._training_table_name].head(2).to_dict(orient='records'))
#
#             print("training table head as dict")
#             print(all_tables_df[self._input_table_name].head(2).to_dict(orient='records'))
#
#             X_train = all_tables_df[self._training_table_name].drop(columns=[self._label_column])
#             y_train = all_tables_df[self._training_table_name][self._label_column]
#             X_test = None
#             y_test = None
#
#             if evaluate:
#                 X_test, X_train, y_test, y_train = self.evaluate_step(X_test, X_train, y_test, y_train)
#             self._x_train_two_rows = X_train.head(2)
#             self._model_input_columns = {column_name: column_index for column_index, column_name in
#                                          enumerate(X_train.columns)}
#
#             input_dim = len(X_train.columns)
#
#             # validate model instance data
#             if list_of_relu_layer_widths is None or len(list_of_relu_layer_widths) == 0 or input_dim == 0:
#                 logging.error("Model instance shape (input_dim: " + str(input_dim) + ", layers: "
#                               + str(list_of_relu_layer_widths) + "), problematic")
#                 return None
#
#             self._model = self.build_model_instance(input_dim, list_of_relu_layer_widths)
#
#             # Compile the model
#             self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#                                 loss='mean_squared_error')
#             if verbose:
#                 print("Model: " + str(self._model))
#             # log the columns of the model
#             logging.info("Model columns: " + str(list(X_train.columns)))
#             self._model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
#             self._is_model_trained = True
#             if evaluate:
#                 self._model.evaluate(X_test, y_test, verbose=2)
#                 # Make predictions on evaluation data
#                 predictions = self._model.predict(X_test)
#                 # Calculate the ratio between actual values and predictions
#                 ratio = y_test / predictions.flatten()
#                 # for each sample, take the max of the two ratios
#                 ratio = ratio.apply(lambda x: max(x, 1 / x))
#                 # cap ratio values at 10
#                 ratio = ratio.apply(lambda x: 10 if x > 10 else x)
#
#                 # combine X_test and y_test into a single dataframe and add the predictions
#                 combined = pd.concat([X_test, y_test], axis=1)
#                 combined['predicted'] = predictions
#
#                 # Visualize the distribution of the ratio
#                 plt.figure(figsize=(10, 6))
#                 sns.histplot(ratio, bins=100, kde=True)
#                 plt.title('Distribution of Ratio (Actual price / Predicted price)')
#                 plt.xlabel('Ratio')
#                 plt.ylabel('Frequency')
#                 plt.show()
#
#             return self._model
#
#     def get_model_input_columns_and_label(self):
#         return self._model_input_columns_and_label
#
#     def evaluate_step(self, X_test, X_train, y_test, y_train):
#         X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
#         print("y_test.index")
#         print(list(y_test.index))
#         print("X_test.index")
#         print(list(X_test.index))
#         print("X_test.columns")
#         print(list(X_test.columns))
#         self._x_train_two_rows = X_train.head(2)
#         return X_test, X_train, y_test, y_train
#
#     def do_all_features_exist(self, columns):
#         cols_difference_set = set(self._all_used_features).difference(set(columns))
#         if len(cols_difference_set) > 0:
#             logging.error("missing columns: " + str(cols_difference_set))
#             return False
#         return True
#
#     def model_predict(self, prepared_row):
#         model_input = prepared_row.astype(int)
#         logging.info("Predicting with input: " + str(model_input))
#         logging.info("Predicting with model: " + str(self._model))
#         return self._model.predict(model_input, verbose=0), model_input
#
#     def build_model_instance(self, input_dim, list_of_relu_layer_widths):
#         model = Sequential()
#         model.add(Dense(list_of_relu_layer_widths[0], activation='relu', input_dim=input_dim))
#         for layer_width in list_of_relu_layer_widths[1:]:
#             model.add(Dense(layer_width, activation='relu'))
#         model.add(Dense(1, activation=self._last_neuron_activation))
#         return model
#
#     # Predict price for a single part based on its features
#     def predict_part_price(self, part_features_dict):
#         logging.info("Predicting part price for: " + str(part_features_dict))
#         row = util_functions.dict_to_df_row(part_features_dict)
#         prepared_row = self.prepare_for_fit_predict(row)
#         logging.info("equivalent of prod: " + str(prepared_row.to_dict(orient='records')) + " " + str(
#             list(prepared_row.columns)))
#         prediction, model_input = self.model_predict(prepared_row)
#         return prediction, model_input, prepared_row, row
#
#     def price_predictions_for_all_feature_combinations(self, all_tables_df, csv_filename=None):
#         print("Started price_predictions_for_all_feature_combinations")
#         # collect all feature values for all features, and then create a cartesian product of all feature values
#         all_values = {feature: all_tables_df[self._input_table_name][feature].unique() for feature in
#                       self._all_part_features}
#
#         combinations = itertools.product(*[all_values[feature] for feature in self._all_part_features])
#
#         ret_df = pd.DataFrame(combinations, columns=self._all_part_features)
#
#         # Predict prices for all feature combinations
#         ret_df['pred_price'] = ret_df.apply(lambda row:
#                                             self.predict_part_price_in_lambda(row),
#                                             axis=1)
#
#         # for element in combinations:
#         #     part_features_dict = {k: [v] for k, v in zip(self._all_part_features, element)}
#         #     predict_rows = self.predict_part_price(part_features_dict)
#         #     predict_rows = pd.DataFrame(predict_rows, columns=['pred_price'])
#         #     predict_rows = pd.concat([pd.DataFrame(part_features_dict), predict_rows], axis=1)
#         #     ret_df = pd.concat([ret_df, predict_rows])
#         if csv_filename is not None:
#             ret_df.to_csv(csv_filename, index=False)
#         return ret_df
#         # display a graph that shows the distribution of prices as a function of tolerance_bucketed
#         # sns.scatterplot(data=ret_df, x='average_tolerance_01_bucketed', y='pred_price')
#
#     def predict_part_price_in_lambda(self, row):
#         prediction, model_input, prepared_row, row = self.predict_part_price(
#             util_functions.dict_to_df_row(row.to_dict()))
#         return prediction
#
#     # Complete columns required for predicting from model (if missing then should be 0 in a 1-hot encoding)
#     def complete_columns_with_negatives(self, prepared_data):
#         if self._model_input_columns_and_label is not None:
#             logging.info(f"New method for completing columns with negatives. Length: {len(prepared_data.columns)}")
#             return prepared_data.reindex(columns=self._model_input_columns_and_label, fill_value=False).drop(
#                 columns=[self._label_column])
#         if self._x_train_two_rows is not None:
#             missing_columns = list(set(self._x_train_two_rows.columns).difference(set(prepared_data.columns)))
#             missing_columns_1 = list(set(self._model_input_columns.keys()).difference(set(prepared_data.columns)))
#             # check if the missing columns are the same as the ones in the model_input_columns
#             if len(missing_columns) != len(missing_columns_1) \
#                     or len(set(missing_columns).symmetric_difference(set(missing_columns_1))) > 0:
#                 logging.error(
#                     "inconsistency in the missing columns : " + str(missing_columns) + " vs " + str(missing_columns_1))
#             else:
#                 logging.error("columns from two rows : " + str(self._x_train_two_rows.columns))
#                 logging.error("columns from model_input_columns : " + str(self._model_input_columns.keys()))
#
#             prepared_data = pd.concat([prepared_data, pd.DataFrame(index=prepared_data.index, columns=missing_columns)],
#                                       axis=1)
#             prepared_data = prepared_data.fillna(0)
#
#             # Reorder (and filter) columns to the order at train time
#             logging.info("Reordering columns to the order at train time")
#             prepared_data = prepared_data[self._x_train_two_rows.columns]
#         else:
#             logging.error("Trying to complete columns for train/predict before knowing the columns.")
#         return prepared_data
#
#     # Feature validation
#     # Only necessary features
#     # One hot encoding
#     def prepare_for_fit_predict(self, input_table_rows_df, verbose=False):
#         # Feature validation
#         if self.do_all_features_exist(input_table_rows_df.columns):
#             ret_df = input_table_rows_df.copy()
#             if verbose:
#                 print("Before one hot encoding: ")
#                 print(len(list(ret_df.columns)))
#                 print(list(ret_df.columns))
#                 print("All training features: ")
#                 print(self._all_training_features)
#                 print("Label column")
#                 print(self._label_column)
#                 print("Categorical: ")
#                 print(self._categorical_features)
#
#             # Only necessary features
#             ret_df = ret_df[ret_df.columns.intersection(self._all_training_features + [self._label_column])]
#             if verbose:
#                 print("ret_df after intersection")
#                 print(len(list(ret_df.columns)))
#                 print(list(ret_df.columns))
#
#             # One hot encoding
#             for categorical_feature in self._categorical_features:
#                 logging.info("One hot encoding for: " + categorical_feature)
#                 if categorical_feature in ret_df.columns:
#                     ret_df = pd.concat([ret_df, pd.get_dummies(ret_df[categorical_feature],
#                                                                prefix=categorical_feature)],
#                                        axis=1).drop(columns=[categorical_feature])
#
#             ret_df = self.complete_columns_with_negatives(prepared_data=ret_df)
#             return ret_df
#         logging.error("Features missing in data passed to prepare_for_fit_predict")
#         raise ValueError("Features missing in data passed to prepare_for_fit_predict")
#
#     @property
#     def model(self):
#         return self._model
