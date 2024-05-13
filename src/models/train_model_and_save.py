# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.models import pred_part_price
from src.models.config import PART_FEATURES_TO_TRAIN_ON, LIST_OF_RELU_LAYER_WIDTHS, BATCH_SIZE, EPOCHS
from src.models.pred_part_price_new import ModelHandler
from src.utils.util_functions import get_all_dataframes_from_parquets


@click.command()
@click.option('--option', default='default')
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
def main(option, input_filepath, model_output_filepath):
    """ Trains model based on processed data and saves to models folder.
    """
    logger = logging.getLogger(__name__)
    if option == 'default':
        logger.info('Running default option - which does nothing')
        train_model_and_save(input_filepath, logger, model_output_filepath)
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


def train_model_and_save(input_filepath, logger, model_output_filepath):
    logger.info('Starting model training...')
    all_tables_df = get_all_dataframes_from_parquets(input_filepath)
    part_price_predictor = pred_part_price.PartPricePredictor(
        list_of_relu_layer_widths=LIST_OF_RELU_LAYER_WIDTHS,
        all_part_features=PART_FEATURES_TO_TRAIN_ON
    )
    part_price_predictor.build_model(all_tables_df=all_tables_df,
                                     list_of_relu_layer_widths=LIST_OF_RELU_LAYER_WIDTHS,
                                     epochs=EPOCHS,
                                     batch_size=BATCH_SIZE,
                                     all_part_features=PART_FEATURES_TO_TRAIN_ON,
                                     evaluate=True,
                                     verbose=True
                                     )
    logger.info('Finished model training.')
    print(part_price_predictor.model)
    # part_price_predictor.save_model(model_output_filepath)
    part_price_predictor.save_model_new()
    logger.info(f'Saved model {part_price_predictor.model_name} to ' + str(model_output_filepath))


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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


