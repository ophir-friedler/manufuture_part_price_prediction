# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.data import dal
from src.data.dal import prices_in_csvs_to_parquets, manufuture_db_to_parquets, save_all_tables_to_parquets, \
    save_all_tables_to_database
from src.data.make_werk_data import werk_to_parquets, process_all_werk_results_dirs_to_df, werk_by_result_name
from src.data.tidy_data import prepare_tidy_data, split_part_price_train_test_tables
from src.models.config import LIST_OF_RELU_LAYER_WIDTHS, PART_FEATURES_TO_TRAIN_ON, PART_FEATURES_PRED_INPUT, \
    WERK_RAW_DICT, EPOCHS, BATCH_SIZE, LEARNING_RATE, PART_DETAILS_JSON
from src.models.part_price_model_serving import ModelServing
from src.models.pred_part_price_new import ModelHandler, drop_all_models


@click.command()
@click.option('--option', default='default')
@click.option('--io', nargs=2, default=None)
@click.option('--mf_data_filepath', nargs=1, default=None)
@click.option('--mf_prices_filepath', nargs=1, default=None)
@click.option('--werk_input_filepath', nargs=1, default=None)
@click.option('--output_filepath', nargs=1, default=None)
@click.option('--model_output_filepath', nargs=1, default=None)
@click.option('--model_name', nargs=1, default=None)
@click.option('--parquet_path', nargs=1, default=None)
def main(option, io, mf_data_filepath=None, mf_prices_filepath=None, werk_input_filepath=None, output_filepath=None,
         model_output_filepath=None, model_name=None, parquet_path=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    if option == 'default':
        logger.info('Running default option - which does nothing')
    if option == 'prices_in_csvs_to_parquets':
        input_filepath, output_filepath = io
        prices_in_csvs_to_parquets(input_filepath, output_filepath)
    if option == 'manufuture_db_to_parquets':
        output_filepath = io[1]
        manufuture_db_to_parquets(output_filepath)
    if option == 'process_werk_data':
        input_filepath, output_filepath = io
        werk_df = process_all_werk_results_dirs_to_df(input_filepath)
        dal.dataframe_to_table(table_name='werk', table_df=werk_df)
        werk_to_parquets(input_filepath, output_filepath)
        dal.dataframe_to_table(table_name='werk_by_name', table_df=werk_by_result_name(werk_df))
    if option == 'prepare_mysql':
        dal.prepare_mysql()
    if option == 'prepare_tidy_data':
        all_tables_df = prepare_tidy_data(mf_data_filepath, mf_prices_filepath, werk_input_filepath)
        logging.info("Done processing all tables")
        save_all_tables_to_parquets(all_tables_df, output_filepath)
        save_all_tables_to_database(all_tables_df)
        split_part_price_train_test_tables()
    if option == 'train_model_and_save':
        model_handler = ModelHandler.get_trained_model(list_of_relu_layer_widths=LIST_OF_RELU_LAYER_WIDTHS,
                                                       epochs=EPOCHS,
                                                       batch_size=BATCH_SIZE,
                                                       learning_rate=LEARNING_RATE,
                                                       target_column_name='unit_price',
                                                       last_neuron_activation='linear',
                                                       loss_function='mean_squared_error',
                                                       categorical_features=PART_FEATURES_TO_TRAIN_ON,
                                                       training_table_name='part_price_training_table_80')
        # Write categorical features to logger:
        logger.info(f'Model categorical_features: {model_handler.categorical_features_dict}')
        model_handler.save_model()
        logger.info(f'Saved model {model_handler.model_name} to ' + str(model_output_filepath))
    if option == 'drop_all_models':
        drop_all_models()
    if option == 'load_model_and_predict':
        model_serving = ModelServing.load_model_by_name(model_name)
        logging.info(f'loaded model: {model_serving.model_name}')
        prediction, model_input = model_serving.predict_part_price(part_features_dict=PART_FEATURES_PRED_INPUT)
        logging.info(f'Finished model prediction: {prediction} on example: {model_input}')
    if option == 'load_model_and_predict_json':
        model_serving = ModelServing.load_model_by_name(model_name)
        logging.info(f'loaded model: {model_serving.model_name}')
        # PART_DETAILS_JSON is a json dict. Translate it to string for the function call
        item_data_json = json.dumps(PART_DETAILS_JSON)
        prediction, model_input = model_serving.predict_on_part_measures(part_data=item_data_json)
        logging.info(f'Finished model prediction: {prediction} on example: {model_input}')

    if option == 'load_model_and_show_inputs':
        model_serving = ModelServing.load_model_by_name(model_name)
        logging.info(f'loaded model: {model_serving.model_name}')
        logging.info(f' Model input dictionary: ' + str(model_serving.pretty_categorical_features_dict()))
    if option == 'load_model_and_predict_on_raw':
        model_handler = ModelHandler.load_model_by_name(model_name)
        logging.info(f'loaded model: {model_handler.model_name}')
        prediction, model_input = model_handler.predict_on_werk_raw_data(werk_raw_dict=WERK_RAW_DICT)
        logging.info(f'Finished model prediction: {prediction} on example: {model_input}')
    if option == 'read_parquet':
        print(dal.read_parquet_into_dataframe(parquet_path))

    # TODO: Create the equivalent of evaluate here
    if option == 'evaluate_model':
        model_handler = ModelHandler.load_model_by_name(model_name)
        model_handler.evaluate_model(evaluation_table_name='part_price_training_table_20')
    if option == 'show_model_details':
        model_handler = ModelHandler.load_model_by_name(model_name)
        model_handler.show_model_details()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
