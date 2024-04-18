# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.models.pred_part_price import PartPricePredictor


@click.command()
@click.option('--model_name', nargs=1, default=None)
def main(model_name):
    """ Trains model based on processed data and saves to models folder.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Starting load model and predict for model: {model_name}')
    model_handler = PartPricePredictor.load_model_by_name(model_name)
    logger.info(f'loaded model: {model_handler.get_model_name()}')
    print(model_handler.predict_part_price(part_features_dict={'max_enclosing_cuboid_volume_bucketed': '[16384-32768)',
                                                               'average_tolerance_01_bucketed': '[0-1)',
                                                               'average_tolerance_001_bucketed': '[0-1)',
                                                               'average_tolerance_0001_bucketed': '[0-1)',
                                                               'average_number_of_nominal_sizes_bucketed': '[16-32)',
                                                               'first_material_categorization_level_1_set': '[NONFERROUS_ALLOY]',
                                                               'first_material_categorization_level_2_set': '[PLATINUM]',
                                                               'quantity_bucket': '[4-5)',
                                                               'machining_process': 'milling'}))
    logger.info('Finished model prediction on example.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
