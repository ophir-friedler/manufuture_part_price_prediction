# -*- coding: utf-8 -*-
import logging

from src.models.config import PART_FEATURES_DICT
from src.models.pred_part_price import PartPricePredictor


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
