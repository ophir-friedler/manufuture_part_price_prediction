# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.data import dal
from src.data.dal import prices_in_csvs_to_parquets, manufuture_db_to_parquets, save_all_tables_to_parquets, \
    save_all_tables_to_database
from src.data.make_werk_data import werk_to_parquets
from src.data.tidy_data import prepare_tidy_data
from src.models.config import LIST_OF_RELU_LAYER_WIDTHS, PART_FEATURES_TO_TRAIN_ON, PART_FEATURES_DICT
from src.models.load_model_and_predict import load_model_and_predict
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
def main(option, io, mf_data_filepath=None, mf_prices_filepath=None, werk_input_filepath=None, output_filepath=None,
         model_output_filepath=None, model_name=None):
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
    if option == 'werk_to_parquets':
        input_filepath, output_filepath = io
        werk_to_parquets(input_filepath, output_filepath)
    if option == 'prepare_mysql':
        dal.prepare_mysql()
    if option == 'prepare_tidy_data':
        all_tables_df = prepare_tidy_data(mf_data_filepath, mf_prices_filepath, werk_input_filepath)
        logging.info("Done processing all tables")
        save_all_tables_to_parquets(all_tables_df, output_filepath)
        save_all_tables_to_database(all_tables_df)
    if option == 'new_flow':
        model_handler = ModelHandler.get_trained_model(list_of_relu_layer_widths=LIST_OF_RELU_LAYER_WIDTHS,
                                                       target_column_name='unit_price',
                                                       last_neuron_activation='linear',
                                                       loss_function='mean_squared_error',
                                                       categorical_features=PART_FEATURES_TO_TRAIN_ON,
                                                       training_table_name='part_price_training_table_646')
        model_handler.save_model()
        logger.info(f'Saved model {model_handler.model_name} to ' + str(model_output_filepath))
    if option == 'drop_all_models':
        drop_all_models()
    if option == 'load_model_and_predict':
        model_handler = ModelHandler.load_model_by_name(model_name)
        logging.info(f'loaded model: {model_handler.get_model_name()}')
        prediction, model_input = model_handler.predict_part_price(part_features_dict=PART_FEATURES_DICT)
        logging.info(f'Finished model prediction: {prediction} on example: {model_input}')
    # TODO: Create the equivalent of evaluate here


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
