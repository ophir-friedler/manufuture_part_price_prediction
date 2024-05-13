# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.models.config import PART_FEATURES_DICT
from src.models.pred_part_price import PartPricePredictor


@click.command()
@click.option('--model_name', nargs=1, default=None)
def main(model_name):
    """ Trains model based on processed data and saves to models folder.
    """
    logger = logging.getLogger(__name__)
    load_model_and_predict(model_name)


def load_model_and_predict(model_name):
    logging.info(f'Starting load model and predict for model: {model_name}')
    model_handler = PartPricePredictor.load_model_by_name(model_name)
    logging.info(f'loaded model: {model_handler.get_model_name()}')
    logging.info(f'Model columns and label: {model_handler.get_model_input_columns_and_label()}')
    prediction, model_input, prepared_row, row = model_handler.predict_part_price(part_features_dict=PART_FEATURES_DICT)
    # Translate model_input from dataframe to dictionary
    logging.info(f'Prediction: {prediction}')
    logging.info(f"Model input: {model_input.to_dict(orient='records')[0]}")
    logging.info(f'Prepared row: {prepared_row}')
    logging.info(f'Row: {row}')
    logging.info('Finished model prediction on example.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
