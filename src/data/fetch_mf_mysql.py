import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.dal import manufuture_db_to_parquets
from src.utils.util_functions import is_path_empty


def clean_table_and_save(table_name, table_df):
    if table_name == 'wp_posts':
        # Some values in table_df['post_date_gmt'] have a value of '0000-00-00 00:00:00' which is not a valid datetime
        # value. Replace these values with NaT
        table_df['post_date_gmt'] = table_df['post_date_gmt'].replace('0000-00-00 00:00:00', None)
        table_df['post_modified_gmt'] = table_df['post_modified_gmt'].replace('0000-00-00 00:00:00', None)

    return table_df


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Read Manufuture MySQL database and saves it to data/raw as parquet files
    """
    logger = logging.getLogger(__name__)
    logger.info('fetching raw data from Manufuture MySQL database')

    # if output_filepath doe not exist, create it
    Path(output_filepath).mkdir(parents=True, exist_ok=True)

    # Check if output_filepath is empty, and if not, ask user if they want to overwrite it
    if not is_path_empty(output_filepath):
        # return
        overwrite = input("fetch_mf_mysql: Manufuture's raw data directory is not empty. Do you want to overwrite it? (y/n): ")
        if overwrite != 'y':
            print("Exiting without overwriting output directory.")
            return

    manufuture_db_to_parquets(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
