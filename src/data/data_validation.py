import logging
import sys

from src.data import dal


def validate_database(db_name):
    if db_name not in dal.get_database_names():
        logging.error(f"validate_database: {db_name} database does not exist")
        return False
    return True


def validate_tables(db_structure):
    database_name = db_structure['database_name']
    tables = db_structure['tables'].keys()
    for table_name in tables:
        if not dal.table_exists(database_name, table_name):
            logging.error(f"validate_tables: {table_name} table does not exist")
            return False
        expected_columns = db_structure['tables'][table_name]['columns'].keys()
        actual_columns = dal.get_table_columns(database_name, table_name)
        if set(expected_columns) != set(actual_columns):
            logging.error(f"validate_tables: {table_name} table does not have the expected columns")
            return False
    return True


def basic_validate(db_name, db_structure):
    if not validate_database(db_name):
        logging.error("Database validation failed")
        return False
    if not validate_tables(db_structure):
        logging.error("Table validation failed")
        return False
    logging.info("Database and tables are valid")
    return True


def data_validate(db_name, db_structure):
    if not validate_database(db_name):
        logging.error("Database validation failed")
        sys.exit(1)
    if not validate_tables(db_structure):
        logging.error("Table validation failed")
        sys.exit(1)
    logging.info("data_validation: Database and tables are valid")
