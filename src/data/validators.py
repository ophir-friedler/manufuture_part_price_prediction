import logging


# Validate that every project has at most 1 quote assigned to it
def validate_quote_to_single_project(all_tables_df):
    wp_type_quote = all_tables_df['wp_type_quote']
    if max(wp_type_quote.groupby(['project'])[['post_id']].count()['post_id']) > 1:
        logging.error("some project has more than 1 quote")


def validate_existence(all_tables_df: dict, table_name_list: list):
    for table_name in table_name_list:
        if table_name not in all_tables_df:
            logging.error("table " + table_name + " is missing")
            raise Exception("table " + table_name + " is missing")