import logging
import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from mysql.connector import Error
from sqlalchemy import create_engine

from src.config import DB_STRUCTURE, DB_NAME, READ_ONLY_DB_NAME
from src.data import data_validation
from src.data.config import SKIPPED_RAW_MANUFUTURE_TABLES, TABLES_TO_SAVE_TO_DB


def connect(database_name=None, verbose=False):
    load_dotenv(find_dotenv())
    mysql_host = os.environ.get('MYSQL_HOST')
    mysql_user = os.environ.get('MYSQL_USER')
    mysql_pw = os.environ.get('MYSQL_PASSWORD')
    connection = mysql.connector.connect(host=mysql_host,
                                         database=database_name,
                                         user=mysql_user,
                                         password=mysql_pw
                                         )
    if connection.is_connected():
        logging.info("connect: Connected to MySQL database") if verbose else None
    cursor = connection.cursor()
    return connection, cursor


def _get_sql_engine(database_name=DB_NAME):
    load_dotenv(find_dotenv())
    mysql_pw = os.environ.get('MYSQL_PASSWORD')
    db_connection_string = f"mysql+pymysql://root:{mysql_pw}@localhost/" + database_name
    sql_engine = create_engine(db_connection_string)  # , pool_recycle=3600
    return db_connection_string, sql_engine


def read_query(query_str: str, database_name=DB_NAME) -> pd.DataFrame:
    try:
        connection, cursor = connect(database_name)
        cursor.execute(query_str)
        result = cursor.fetchall()
        result = pd.DataFrame(result, columns=[column[0] for column in cursor.description])
        cursor.close()
        connection.close()
        return result
    except Error as e:
        logging.error(f"Error (read_query_direct): {e}")
    return pd.DataFrame()


def read_table_into_dataframe(table_name, database_name=DB_NAME) -> pd.DataFrame:
    db_connection_string, sql_engine = _get_sql_engine(database_name)
    try:
        db_connection = sql_engine.connect()
        query = f'SELECT * FROM `{table_name}`'
        ret_val = pd.read_sql(query, db_connection)
        db_connection.close()
        return ret_val
    except Error as e:
        logging.error(f"Error while connecting, connection string: {db_connection_string}")
        logging.error(f"Error while connecting to MySQL: {e}")
    return pd.DataFrame()


def read_parquet_into_dataframe(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def get_distinct_values(table_name, column_name):
    db_connection_string, sql_engine = _get_sql_engine()
    try:
        db_connection = sql_engine.connect()
        query = f'SELECT DISTINCT {column_name} FROM `{table_name}`'
        ret_val = pd.read_sql(query, db_connection)
        db_connection.close()
        return ret_val
    except Error as e:
        logging.error(f"Error while connecting, connection string: {db_connection_string}")
        logging.error(f"Error while connecting to MySQL: {e}")
    return pd.DataFrame()


def get_unique_primary_key_values(table_name, primary_key_columns: list, handled_date=None):
    # report_str = f"Getting unique primary key values for table {table_name}"
    # if handled_date is not None:
    #     report_str += f" for date {handled_date}"
    # logging.info(report_str)
    db_connection_string, sql_engine = _get_sql_engine()
    try:
        if not table_exists(DB_NAME, table_name):
            return pd.DataFrame()
        db_connection = sql_engine.connect()
        query = f'SELECT DISTINCT {",".join(primary_key_columns)} FROM `{table_name}`'
        if handled_date is not None:
            query += f" WHERE date = '{handled_date}'"
        ret_val = pd.read_sql(query, db_connection)
        db_connection.close()
        return ret_val
    except Error as e:
        logging.error(f"Error while connecting, connection string: {db_connection_string}")
        logging.error(f"Error while connecting to MySQL: {e}")
    return pd.DataFrame()


def create_table(database_name, table_name, dict_columns_name_to_type, primary_key=None):
    try:
        connection, cursor = connect(database_name)
        query_str = ''
        try:
            column_name_to_type_list = ','.join(
                [f'`{column_name}` {column_type}' for column_name, column_type in dict_columns_name_to_type.items()])
            query_str = f"CREATE TABLE {table_name} ({column_name_to_type_list}" \
                        + (f", PRIMARY KEY ({','.join(primary_key)}))" if primary_key is not None else ')')
            logging.info(f"Creating table {table_name} with query: {query_str}")
            cursor.execute(query_str)

            logging.info(f"Table {table_name} created successfully")
        except Error as e:
            logging.info(f"Error (create_table): {e}, query: {query_str}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        logging.info(f"Error while connecting to MySQL: {e}")


def drop_table(database_name, table_name, verbose=False):
    try:
        connection, cursor = connect(database_name, verbose=verbose)
        try:
            cursor.execute(f"DROP TABLE {table_name}")
            logging.info(f"Table {table_name} dropped successfully") if verbose else None
        except Error as e:
            logging.error(f"Error (drop_table): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")


# Get a dataframe, and add rows to an existing mysql table according to the dataframe's name and columns
def append_dataframe_to_table(table_name, table_df, handled_date=None, verbose=False):
    try:
        if handled_date is not None:
            table_df = table_df[table_df['date'] == handled_date]

        _, sql_engine = _get_sql_engine()
        if table_df.empty:
            report_str = f"Nothing to add to {table_name}."
            report_str += f" for date {handled_date}" if handled_date is not None else ''
            logging.info(report_str) if verbose else None
            return

        primary_key_columns = DB_STRUCTURE['tables'][table_name]['primary_key']
        temp_table_name = 'temp_table_' + str(int(pd.Timestamp.now().timestamp()))

        dataframe_to_table(temp_table_name, table_df)
        query_str = f"SELECT {temp_table_name}.* FROM {temp_table_name} " \
                    f"LEFT JOIN {table_name} ON "
        query_str += " AND ".join(
            [f"{temp_table_name}.{column} = {table_name}.{column}" for column in primary_key_columns])
        query_str += f" WHERE {table_name}.{primary_key_columns[0]} IS NULL"
        query_str += f" AND date = '{handled_date}' " if handled_date is not None else ''
        result_df = read_query(query_str=query_str)
        if result_df.empty:
            report_str = f"Table {table_name} is up to date"
            report_str += f" for date {handled_date}" if handled_date is not None else ''
            logging.info(report_str) if verbose else None
            drop_table(DB_NAME, temp_table_name)
            return
        report_str = f"Appending to table {table_name}"
        report_str += f" for date {handled_date}" if handled_date is not None else ''
        logging.info(report_str) if verbose else None
        result_df.to_sql(table_name, con=sql_engine, if_exists='append', index=False)
        drop_table(DB_NAME, temp_table_name)
        logging.info(f"Added {result_df.shape[0]} rows to table {table_name}") if verbose else None
    except Error as e:
        logging.error(f"Error (append_dataframe_to_table): {e}") if verbose else None


def dataframe_to_table(table_name, table_df, verbose=False):
    try:
        _, sql_engine = _get_sql_engine()
        logging.info(f" Writing to {table_name}") if verbose else None
        table_df.to_sql(table_name, con=sql_engine, if_exists='replace', index=False)
        logging.info(f"Table {table_name} (over?) written successfully") if verbose else None
    except Error as e:
        logging.error(f"Error (dataframe_to_mysql_table): {e}")


def drop_database(database_name, verbose=False):
    try:
        connection, cursor = connect(database_name, verbose=verbose)
        try:
            cursor.execute(f"DROP DATABASE {database_name}")

            logging.info(f"Database {database_name} dropped successfully")
        except mysql.connector.Error as e:
            logging.error(f"Error (drop_database): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")


def get_database_names(verbose=False):
    try:
        connection, cursor = connect(database_name=None, verbose=verbose)
        if connection.is_connected():
            logging.info("get_database_names: Connected to MySQL database") if verbose else None

        cursor = connection.cursor()
        try:
            cursor.execute("SHOW DATABASES")
            database_names = cursor.fetchall()
            database_names = [database_name[0] for database_name in database_names]
            cursor.close()
            connection.close()
            return database_names
        except mysql.connector.Error as e:
            logging.error(f"Error (get_database_names): {e}")
        cursor.close()
        connection.close()
    except mysql.connector.Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")


def create_database(database_name, verbose=False):
    try:
        connection, cursor = connect(database_name=None, verbose=verbose)
        try:
            cursor.execute(f"CREATE DATABASE {database_name}")

            logging.info(f"Database {database_name} created successfully")
        except mysql.connector.Error as e:
            logging.info(f"Error (create_project_database): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as e:
        logging.info(f"Error while connecting to MySQL: {e}")


def delete_rows_from_table(database_name, table_name, column_name, column_value):
    try:
        connection, cursor = connect(database_name)
        try:
            cursor.execute(f"DELETE FROM {table_name} WHERE {column_name} = '{column_value}'")
            logging.info(f"Rows with {column_name} = {column_value} were deleted from {table_name}")
        except Error as e:
            logging.error(f"Error (delete_rows_from_table): {e}")
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")


def table_exists(database_name, table_name) -> bool:
    try:
        connection, cursor = connect(database_name)
        try:
            cursor.execute("SHOW TABLES")
            result = cursor.fetchall()
            cursor.close()
            connection.close()
            return table_name in [table[0] for table in result]
        except Error as e:
            logging.error(f"Error (table_exists): {e}")
        cursor.close()
        connection.close()
    except Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")


def get_table_names(database_name) -> list:
    try:
        connection, cursor = connect(database_name)
        try:
            cursor.execute("SHOW TABLES")
            result = cursor.fetchall()
            cursor.close()
            connection.close()
            return [table[0] for table in result]
        except Error as e:
            logging.error(f"Error (get_all_tables): {e}")
        cursor.close()
        connection.close()
    except Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")


def get_table_columns(database_name, table_name) -> list:
    try:
        connection, cursor = connect(database_name)
        try:
            cursor.execute(f"SHOW COLUMNS FROM {table_name}")
            result = cursor.fetchall()
            cursor.close()
            connection.close()
            return [column[0] for column in result]
        except Error as e:
            logging.error(f"Error (get_table_columns): {e}")
        cursor.close()
        connection.close()
    except Error as e:
        logging.error(f"Error while connecting to MySQL: {e}")


def is_table_empty(database_name, table_name):
    try:
        connection, cursor = connect(database_name)
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cursor.fetchall()
            cursor.close()
            connection.close()
            return result[0][0] == 0
        except Error as e:
            logging.error(f"Error (in is_table_empty): {e}")
        cursor.close()
        connection.close()
    except Error as e:
        logging.error(f"Error (in is_table_empty) while connecting to MySQL: {e}")
    return True


def prepare_mysql(ask_to_overwrite=True):
    logger = logging.getLogger(__name__)
    logger.info('Preparing environment (MySQL)')
    if ask_to_overwrite:
        if DB_NAME in get_database_names():
            overwrite = input(
                "prepare: " + DB_NAME +
                " database already exists. Do you want to overwrite it? (y/n): ")
            if overwrite != 'y':
                print("Not overwriting existing tables in " + DB_NAME + " database.")
            else:
                drop_database(DB_STRUCTURE['database_name'])
    create_db_and_tables()
    data_validation.basic_validate(DB_NAME, DB_STRUCTURE)


def create_db_and_tables():
    # check if database exists, if not create it
    if DB_NAME not in get_database_names():
        create_database(DB_NAME)
    for table_name in DB_STRUCTURE['tables']:
        if not table_exists(DB_NAME, table_name):
            create_table(DB_NAME,
                         table_name,
                         DB_STRUCTURE['tables'][table_name]['columns'],
                         DB_STRUCTURE['tables'][table_name]['primary_key'],
                         )


def fetch_all_tables_df():
    logging.info('fetching raw data from Manufuture MySQL database')
    table_names_list = get_table_names(READ_ONLY_DB_NAME)
    all_tables_df = {}
    for table_name in table_names_list:
        all_tables_df[table_name] = read_table_into_dataframe(table_name, READ_ONLY_DB_NAME)

    # Load e-mail logs to all_tables_df['email_logs']:
    # all_tables_df['email_logs'] = pd.read_csv(EMAIL_LOGS_DIR)
    return all_tables_df


def mysql_table_to_dataframe(table_name, db_connection) -> pd.DataFrame:
    return pd.read_sql('SELECT * FROM `' + table_name + '`', db_connection)


def prices_in_csvs_to_parquets(input_filepath, output_filepath):
    raw_csv_to_df = {}
    for csv_file in Path(input_filepath).iterdir():
        if csv_file.suffix != '.csv':
            continue
        table_name = csv_file.stem
        logging.info('Reading table ' + table_name + ' from ' + str(csv_file))
        raw_csv_to_df[table_name] = pd.read_csv(csv_file)
    # Save dataframes to parquet files
    for table_name, table_df in raw_csv_to_df.items():
        table_df.to_parquet(output_filepath + '/' + table_name + '.parquet')
        logging.info('Saved table ' + table_name + ' to ' + output_filepath + '/' + table_name + '.parquet')


def manufuture_db_to_parquets(output_filepath):
    for table_name, table_df in fetch_all_tables_df().items():
        if table_name in SKIPPED_RAW_MANUFUTURE_TABLES:
            continue
        if table_name == 'wp_posts':
            table_df['post_date_gmt'] = table_df['post_date_gmt'].replace('0000-00-00 00:00:00', None)
            table_df['post_modified_gmt'] = table_df['post_modified_gmt'].replace('0000-00-00 00:00:00', None)
        print("Writing table " + table_name + " to " + output_filepath + "/" + table_name + ".parquet")
        # validate that output_filepath exists, and if not, create it
        Path(output_filepath).mkdir(parents=True, exist_ok=True)
        table_df.to_parquet(output_filepath + "/" + table_name + ".parquet")


def save_all_tables_to_parquets(all_tables_df, output_filepath):
    # Save all tables to parquet files in output_filepath
    for table_name, table_df in all_tables_df.items():
        print("Writing table " + table_name + " to " + output_filepath + "/" + table_name + ".parquet")
        # validate that output_filepath exists, and if not, create it
        Path(output_filepath).mkdir(parents=True, exist_ok=True)
        table_df.to_parquet(output_filepath + "/" + table_name + ".parquet")


def save_all_tables_to_database(all_tables_df):
    logging.info("Saving all_tables_df to database")
    logging.info("all_tables_df keys: " + str(all_tables_df.keys()))
    for table_name, table_df in all_tables_df.items():
        if table_name in TABLES_TO_SAVE_TO_DB:
            logging.info("Writing table " + table_name + " to database")
            # logging.info("table_df columns: " + str(list(table_df.columns)))
            # logging.info("table_df column types: " + str(list(table_df.dtypes)))
            if table_name in DB_STRUCTURE['tables']:
                table_df_to_save = table_df[DB_STRUCTURE['tables'][table_name]['columns'].keys()]
            else:
                table_df_to_save = table_df
            dataframe_to_table(table_name=table_name,
                               table_df=table_df_to_save)
