import logging
from pathlib import Path

import pandas as pd
from IPython.core.display_functions import display

TABLES_TO_IGNORE_IN_SEARCH = ['wp_quotes', 'pm_project_manufacturer', 'pam_project_active_manufacturer_th_1']


def display_and_return_is_empty(ret_df, print_str):
    is_empty = True
    if len(ret_df) > 0:
        is_empty = False
        print(print_str)
        display(ret_df)
    return is_empty


def searchPostId(post_id, all_tables_df):
    print("All entries for post_id: " + str(post_id))
    is_empty = True
    for table_name, table_df in all_tables_df.items():
        if table_name in TABLES_TO_IGNORE_IN_SEARCH:
            continue
        for post_id_col in ['post_id', 'agency', 'project', 'ID']:
            if post_id_col in table_df.columns:
                is_empty = display_and_return_is_empty(table_df[table_df[post_id_col] == post_id],
                                                       table_name + ": (in column '" + post_id_col + "')") and is_empty

        for post_id_list_col in ['competing_manufacturers', 'bids', 'chosen_bids']:
            if post_id_list_col in table_df.columns:
                is_empty = display_and_return_is_empty(table_df[table_df[post_id_list_col].apply(lambda x: post_id in x)],
                                                       table_name + ": (in column '" + post_id_list_col + "')") and is_empty
    if is_empty:
        print("Error: Did not find any data on post id: " + str(post_id))
        return "EMPTY"


def searchString(str_val, all_tables_df):
    for table_name, table_df in all_tables_df.items():
        if table_name in TABLES_TO_IGNORE_IN_SEARCH:
            continue
        # print(table_name)
        for colname in list(table_df.columns):
            found_val = False
            for val in [str(val) for val in all_tables_df[table_name][colname].astype(str).unique()]:
                if str_val in val:
                    print("table: " + str(table_name))
                    print("   colname: " + str(colname))
                    print("   value: " + str(val))
                    found_val = True
            if found_val:
                display(all_tables_df[table_name][all_tables_df[table_name][colname].astype(str).str.find(str_val) != -1])


def searchForColumn(str_val, all_tables_df):
    for table_name, table_df in all_tables_df.items():
        # if table_name in TABLES_TO_IGNORE_IN_SEARCH:
        #     continue
        # print(table_name)
        for colname in list(table_df.columns):
            if str_val in str(colname):
                print("table: " + str(table_name))
                print("   colname: " + str(colname))


def parse_list_of_integers(list_of_integers_str):
    if list_of_integers_str is None or len(list_of_integers_str.strip()) == 0:
        return []
    if list_of_integers_str.isdigit():
        return [int(list_of_integers_str)]
    return [int(integer) for integer in list_of_integers_str[1:-1].split(",")]


# extend list_of_bids to contain also bid ids that are in integer_or_list
def extend_list_of_bids(list_of_bids, integer_or_list):
    if isinstance(integer_or_list, list):
        list_of_bids.extend(integer_or_list)
        return
    if len(integer_or_list) == 0:
        logging.warning("Warning: found an empty entry of bids.")
        return
    if integer_or_list.isdigit():
        list_of_bids.append(integer_or_list)
    else:
        # list format [1, 2, 3], remove brackets
        all_bids_in_list = integer_or_list[1:-1].split(",")
        list_of_bids.extend(all_bids_in_list)


# build all_bid_ids: list of post ids of bids (taken from wp_type_quote):
def get_all_bid_ids(all_tables_df):
    all_bid_ids = []
    all_tables_df['wp_type_quote']['bids'].apply(lambda bids: extend_list_of_bids(all_bid_ids, bids))
    all_bid_ids = set(map(int, all_bid_ids))
    return all_bid_ids


def diff_days_or_none(start_date, end_date):
    # print(str(type(start_date)) + " ,  " + str(type(end_date)))
    if (pd.isnull(start_date)) or (pd.isnull(end_date)):
        #         print("row is None")
        return None
    #     print("row is NOT None >>" + str(tmp) + "<<" )
    return (end_date - start_date).days


def add_columns_year_month_day_from_datetime(df, columns_prefix, datetime_column):
    return df.apply(lambda row: add_date_time_columns(row, columns_prefix, datetime_column), axis=1)


def add_date_time_columns(row, columns_prefix, datetime_column):
    if row[datetime_column] is pd.NaT:
        row[columns_prefix + '_year'] = 0
        row[columns_prefix + '_month'] = 0
        row[columns_prefix + '_day'] = 0
        row[columns_prefix + '_year_month'] = "0-0"
        row[columns_prefix + '_Ym'] = "0-0"
    else:
        row[columns_prefix + '_year'] = row[datetime_column].year  # .astype(int)
        row[columns_prefix + '_month'] = row[datetime_column].month  # .astype(int)
        row[columns_prefix + '_day'] = row[datetime_column].day  # .astype(int)
        row[columns_prefix + '_year_month'] = (str(row[columns_prefix + '_year']) + "-" + str(row[columns_prefix + '_month']))
        row[columns_prefix + '_Ym'] = pd.to_datetime(row[columns_prefix + '_year_month'], format='%Y-%m').strftime('%Y-%m')
    return row


def convert_str_set_to_float_set(str_set):
    # if str_set is not string, throw exception
    if not isinstance(str_set, str):
        raise Exception("Error: str_set is not string: " + str(str_set))
    if str_set is None or len(str_set) < 2:
        return None
    try:
        elements = str_set[1:-1].split(',')
        if elements == ['']:
            return set()
        return set(map(float, str_set[1:-1].split(',')))
    except ValueError:
        logging.warning("Error: could not convert str set to float set: " + str(str_set))
        return None


def get_all_dataframes_from_parquets(path):
    all_tables_df = {}
    for parquet_file in Path(path).glob('**/*.parquet'):
        # set the table name to be the relative path from path to parquet_file without the .parquet suffix
        table_name = str(parquet_file.relative_to(path)).replace('.parquet', '')
        print("Reading table " + table_name + " from " + str(parquet_file))
        all_tables_df[table_name] = pd.read_parquet(parquet_file)
    return all_tables_df


# def dict_to_df_row(dict_):
#     return pd.DataFrame({k: [v] for k, v in dict_.items()})
#
# def get_all_dataframes_in_data_dirs():
#     return get_all_dataframes_from_parquets(os.environ.get("PYTHONPATH") + '/data')
#
# def is_path_empty(path):
#     if not Path(path).exists():
#         return False
#     # Check if output_filepath is empty, and if not, ask user if they want to overwrite it.
#     files_list = list(Path(path).iterdir())
#     # Ignore .gitkeep and .DS_Store files
#     files_list = [file for file in files_list if file.name not in ['.gitkeep', '.DS_Store']]
#     if len(files_list) > 0:
#         return False
#     return True
#
#
# def string_to_hex(string):
#     # Initialize sum to 0
#     total_sum = 0
#
#     # Iterate over each character in the string
#     for char in string:
#         # Convert the character to its binary representation
#         binary_value = bin(ord(char))[2:]
#
#         # Convert binary value to integer and add to the total sum
#         total_sum += int(binary_value, 2)
#
#     # Convert total sum to hexadecimal and return
#     return hex(total_sum)
