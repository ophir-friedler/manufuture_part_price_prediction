import sys

import pandas as pd

from src.data.dal import _get_sql_engine

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes basic tests!")

    db_connection = _get_sql_engine().connect()
    if db_connection is not None:
        print(">>> MySQL connection established!")
    all_table_names = pd.read_sql(f'SHOW TABLES', db_connection)['Tables_in_manufuture']
    print(">>> Tables in Manufuture MySQL database: " + str(all_table_names.keys()))


if __name__ == '__main__':
    main()
