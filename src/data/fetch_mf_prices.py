import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
import pandas as pd

from src.utils.util_functions import is_path_empty


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Read Manufuture Prices from external csv files and save dataframe to data/interim as parquet files
    """
    logger = logging.getLogger(__name__)
    logger.info('fetching Manufuture prices from external data')

    # if output_filepath doe not exist, create it
    Path(output_filepath).mkdir(parents=True, exist_ok=True)

    if not is_path_empty(output_filepath):
        overwrite = input("fetch_mf_prices: Manufuture's raw prices directory is not empty. Do you want to overwrite it? (y/n): ")
        if overwrite != 'y':
            print("Exiting without overwriting output directory.")
            return

    # Read csv files from input_path and save them to dataframes
    raw_csv_to_df = {}
    for csv_file in Path(input_filepath).iterdir():
        table_name = csv_file.stem
        raw_csv_to_df[table_name] = pd.read_csv(csv_file)

    # Save dataframes to parquet files
    for table_name, table_df in raw_csv_to_df.items():
        table_df.to_parquet(output_filepath + '/' + table_name + '.parquet')
        logger.info('Saved table ' + table_name + ' to ' + output_filepath + '/' + table_name + '.parquet')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
