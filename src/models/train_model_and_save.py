# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.models import pred_part_price
from src.models.config import all_part_features, LIST_OF_RELU_LAYER_WIDTHS, BATCH_SIZE, EPOCHS
from src.utils.util_functions import get_all_dataframes_from_parquets


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
def main(input_filepath, model_output_filepath):
    """ Trains model based on processed data and saves to models folder.
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting model training...')
    all_tables_df = get_all_dataframes_from_parquets(input_filepath)
    part_price_predictor = pred_part_price.PartPricePredictor()
    part_price_predictor.build_model(all_tables_df=all_tables_df,
                                     list_of_relu_layer_widths=LIST_OF_RELU_LAYER_WIDTHS,
                                     epochs=EPOCHS,
                                     batch_size=BATCH_SIZE,
                                     all_part_features=all_part_features,
                                     evaluate=True,
                                     verbose=True
                                     )
    logger.info('Finished model training.')
    print(part_price_predictor.model)
    # part_price_predictor.save_model(model_output_filepath)
    part_price_predictor.save_model_new()
    logger.info('Saved model to ' + str(model_output_filepath))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
