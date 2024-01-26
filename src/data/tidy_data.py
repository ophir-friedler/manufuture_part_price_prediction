import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from phpserialize import dict_to_list, loads
import math

from src.data import enrichers, aggregators, validators
from src.data.config import MANUFACTURER_BID_LABEL_COLUMN_NAME, MIN_NUM_BIDS_PER_MANUFACTURER, COUNTRY_TO_ISO_MAP
from src.utils.util_functions import get_all_dataframes_from_parquets, is_path_empty


@click.command()
@click.argument('mf_input_filepath', type=click.Path(exists=True))
@click.argument('werk_input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(mf_input_filepath, werk_input_filepath, output_filepath):
    """ Reads Manufuture and Werk data from parquet files, tidies it, and saves it to output_filepath as parquet files
    """
    if not is_path_empty(output_filepath):
        overwrite = input("tidy_data: The processed data directory is not empty. Do you want to overwrite it? (y/n): ")
        if overwrite != 'y':
            print("Exiting without overwriting output directory.")
            return

    logger = logging.getLogger(__name__)
    logger.info('fetching parquets from Manufuture and Werk')
    # Read all parquet files from path, and save them to all_tables_df
    all_tables_df = get_all_dataframes_from_parquets(mf_input_filepath)
    all_tables_df.update(get_all_dataframes_from_parquets(werk_input_filepath))
    prepare_all_tidy_tables(all_tables_df)
    save_all_tables_to_parquets(all_tables_df, output_filepath)


def save_all_tables_to_parquets(all_tables_df, output_filepath):
    # Save all tables to parquet files in output_filepath
    for table_name, table_df in all_tables_df.items():
        print("Writing table " + table_name + " to " + output_filepath + "/" + table_name + ".parquet")
        # validate that output_filepath exists, and if not, create it
        Path(output_filepath).mkdir(parents=True, exist_ok=True)
        table_df.to_parquet(output_filepath + "/" + table_name + ".parquet")


def prepare_all_tidy_tables(all_tables_df):
    clean_wp_manufacturers(all_tables_df)
    clean_wp_parts(all_tables_df)
    build_wp_tables_by_post_type(all_tables_df)
    build_user_to_entity_rel(all_tables_df)
    build_netsuite_by_memo(all_tables_df)
    enrichers.enrich_all(all_tables_df)
    build_training_data_tables(all_tables_df)
    # Aggregated statistics
    aggregators.monthly_bid_success_rate_df(all_tables_df)
    aggregators.monthly_projects_stats(all_tables_df)
    aggregators.monthly_manufacturers_stats(all_tables_df)
    aggregators.stats_by_num_candidates(all_tables_df)
    validators.validate_quote_to_single_project(all_tables_df)


def build_training_data_tables(all_tables_df):
    logging.info("Building training data tables")
    pm_project_manufacturer(all_tables_df)
    pam_project_active_manufacturer(all_tables_df, 1)
    ac_agency_manufacturer(all_tables_df)
    build_proj_manu_training_table(all_tables_df, MIN_NUM_BIDS_PER_MANUFACTURER)
    build_part_price_training_table(all_tables_df)


def build_part_price_training_table(all_tables_df):
    logging.info("Building part_price_training_table")
    parts_with_netsuite_prices = all_tables_df['wp_type_part'][all_tables_df['wp_type_part']['Rate mean_netsuite'].notnull()]
    parts_with_prices_and_werk = parts_with_netsuite_prices[parts_with_netsuite_prices['found_werk'] == 1]
    # Filter out parts with volume <= 0
    parts_with_prices_and_werk = parts_with_prices_and_werk[parts_with_prices_and_werk['max_enclosing_cuboid_volume'] > 0]
    all_tables_df['part_price_training_table'] = parts_with_prices_and_werk


# Build pam + filter by project requirements
def build_proj_manu_training_table(all_tables_df, min_num_manufacturer_bids):
    pam_th_table_name = pam_project_active_manufacturer(all_tables_df, min_num_manufacturer_bids)
    # pam_th_req_filter_table_name = pam_filter_by_project_requirements(all_tables_df, pam_th_table_name)
    pam_th_req_label_table_name = pam_label_by_project_requirements(all_tables_df, pam_th_table_name)
    logging.warning("proj_manu_training_table_name: " + pam_th_req_label_table_name)
    return pam_th_req_label_table_name


def pam_label_by_project_requirements(all_tables_df, pam_table_name):
    new_table_name: str = pam_table_name + '_label_reqs'
    # print("Writing " + new_table_name)
    # for efficiency
    if new_table_name not in all_tables_df:
        start_table = all_tables_df[pam_table_name]
        # label 0 == 'not bid' all rows where the manufacturer does not have the required capabilities
        start_table[(start_table['cnc_milling'] < start_table['req_milling'])
                    | (start_table['cnc_turning'] < start_table['req_turning'])][MANUFACTURER_BID_LABEL_COLUMN_NAME] = 0
        all_tables_df[new_table_name] = start_table.copy()

    return new_table_name


# Dependencies: pam_project_active_manufacturer
def ac_agency_manufacturer(all_tables_df):
    ac_df = all_tables_df['pm_project_manufacturer'] \
        .reset_index().groupby(['agency', 'post_id_manuf'])[[MANUFACTURER_BID_LABEL_COLUMN_NAME]] \
        .agg({MANUFACTURER_BID_LABEL_COLUMN_NAME: ['sum']})
    ac_df.columns = ['num_bids']
    all_tables_df['ac_agency_manufacturer'] = ac_df


# Dependencies: wp_type_quote (enriched), wp_projects, wp_manufacturers
# Builds pm_project_manufacturer
def pm_project_manufacturer(all_tables_df):
    # training data + labels are based on wp_type_quote
    # Get all project ids that have quotes
    wp_type_quote = all_tables_df['wp_type_quote'][
        ['post_id', 'bids', 'project', 'competing_manufacturers', 'winning_manufacturers']]
    # Join projects features
    wp_projects = all_tables_df['wp_projects']
    pm_df = wp_type_quote.merge(wp_projects, left_on='project', right_on='post_id', suffixes=('_quote', '_project'))
    # For each manufacturer create a project-manufacturer row with data from wp_manufacturers
    wp_manufacturers = all_tables_df['wp_manufacturers'].rename(columns={'post_id': 'post_id_manuf'})
    # TODO: replace wp_manufacturers with wp_type_manufacturer
    pm_df = pm_df.merge(wp_manufacturers, how='cross', suffixes=('_quote', '_manuf'))

    # Clean columns data
    # TODO: move standardize_country_values to write to a new column 'country_iso'
    standardize_country_values(pm_df)

    # build Label column
    pm_df[MANUFACTURER_BID_LABEL_COLUMN_NAME] = pm_df.apply(lambda row:
                                                            1 if row['post_id_manuf'] in row['competing_manufacturers'] else 0, axis='columns')

    # set index by project-manufacturer
    pm_df = pm_df.set_index(['post_id_project', 'post_id_manuf'])
    all_tables_df['pm_project_manufacturer'] = pm_df


def standardize_country_values(training_data):
    if 'country' in training_data.columns:
        training_data['country'] = training_data['country'].transform(
            lambda val: logging.error(": value " + val + " not in country map") if val not in COUNTRY_TO_ISO_MAP.keys()
            else COUNTRY_TO_ISO_MAP[val])
    else:
        logging.warning('country not in training_data.columns')


# Create pam_project_active_manufacturer_th_<num_bids_activation_threshold>
def pam_project_active_manufacturer(all_tables_df, num_bids_activation_threshold):
    new_table_name: str = 'pam_project_active_manufacturer_th_' + str(num_bids_activation_threshold)
    if new_table_name not in all_tables_df:
        pm = all_tables_df['pm_project_manufacturer'].reset_index()
        manuf_num_bids = pm.groupby(['post_id_manuf'])[[MANUFACTURER_BID_LABEL_COLUMN_NAME]].sum().reset_index()
        manuf_num_bids.columns = ['post_id_manuf', 'manuf_num_bids']
        active_manufs = manuf_num_bids[manuf_num_bids['manuf_num_bids'] >= num_bids_activation_threshold]
        pm_only_active_manufs = pd.merge(pm, active_manufs, on='post_id_manuf', how='inner')
        all_tables_df[new_table_name] = pm_only_active_manufs
        logging.info(new_table_name + " created in all_tables_df")

    return new_table_name


def build_netsuite_by_memo(all_tables_df):
    logging.info("Building netsuite_by_memo")
    netsuite_prices_df = pd.read_csv('/Users/ofriedler/Dropbox/Work/Consultation/Manufuture/dev/manufuture/netsuite_prices.csv')
    netsuite_prices_df['average_Rate'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('mean')
    netsuite_prices_df['min_Rate'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('min')
    netsuite_prices_df['max_Rate'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('max')
    netsuite_prices_df['Currencies'] = netsuite_prices_df.groupby('Memo')[['Currency']].agg({'Currency': lambda x: ", ".join(list(x))})
    netsuite_prices_df['num_duplicates'] = netsuite_prices_df.groupby('Memo')['Rate'].transform('count')
    all_tables_df['netsuite_prices'] = netsuite_prices_df

    netsuite_by_memo = netsuite_prices_df.groupby('Memo').agg({'Rate': ['min', 'max', 'count', 'mean'],
                                                               'Quantity': ['min', 'max'],
                                                               'Currency': lambda x: ", ".join(list(x))})
    netsuite_by_memo.columns = [' '.join(col).strip() for col in netsuite_by_memo.columns.values]
    netsuite_by_memo = netsuite_by_memo.reset_index()
    netsuite_by_memo = netsuite_by_memo.add_suffix('_netsuite')
    all_tables_df['netsuite_by_memo'] = netsuite_by_memo


# Dependencies: all_tables_df['wp_usermeta']
# user type (manufacturer, agency), post_id of user type, status (of manufacturer - vendor/pending_vendor)
def build_user_to_entity_rel(all_tables_df):
    logging.info("Building user_to_entity_rel: user_id, user_type, user_type_post_id, user_type_status")

    # group wp_usermeta by user_id and apply the extract_user_info_from_wp_usermeta function
    user_info_df = all_tables_df['wp_usermeta'].groupby('user_id').apply(
        extract_user_info_from_wp_usermeta).reset_index(drop=True)
    user_info_df['user_type_post_id'] = user_info_df['user_type_post_id'].fillna(-1).astype(int)
    all_tables_df['user_to_entity_rel'] = user_info_df


# Return: 'user_id', 'user_type', 'user_type_post_id', 'user_type_status'
def extract_user_info_from_wp_usermeta(user_id_group):
    manufacturer_info = user_id_group[
        (user_id_group['meta_key'] == 'rel_manufacturer') & (user_id_group['meta_value'].str.len() > 0)]
    agency_info = user_id_group[
        (user_id_group['meta_key'] == 'rel_agency') & (user_id_group['meta_value'].str.len() > 0)]
    status_info = user_id_group[user_id_group['meta_key'] == 'wp_capabilities']
    manufacturer_name = manufacturer_info.iloc[0]['meta_value'] if len(manufacturer_info) > 0 else None
    agency_name = agency_info.iloc[0]['meta_value'] if len(agency_info) > 0 else None
    # status_detail = a:1:{s:14:"pending_vendor";b:1;}
    user_id = user_id_group.iloc[0]['user_id']
    result = []
    if manufacturer_name is not None:
        manufacturer_status = ""
        if len(status_info.iloc[0]['meta_value']) > 0 and "\"pending_vendor\"" in status_info.iloc[0]['meta_value']:
            manufacturer_status = 'pending_vendor'
        if len(status_info.iloc[0]['meta_value']) > 0 and "\"vendor\"" in status_info.iloc[0]['meta_value']:
            manufacturer_status = 'vendor'
        result.append((user_id, 'manufacturer', manufacturer_name, manufacturer_status))
    if agency_name is not None:
        result.append((user_id, 'agency', agency_name, None))
    return pd.DataFrame(result, columns=['user_id', 'user_type', 'user_type_post_id', 'user_type_status'])


def build_wp_tables_by_post_type(all_tables_df):
    all_post_types = list(all_tables_df['wp_posts']['post_type'].unique())
    wp_posts = all_tables_df['wp_posts']
    wp_postmeta = all_tables_df['wp_postmeta']
    for post_type in all_post_types:
        print("Building wp_type_" + post_type)
        wp_type_posttype = 'wp_type_' + post_type
        wp_posts_post_type = wp_posts[wp_posts['post_type'] == post_type]
        wp_postmeta_post_type = wp_postmeta[(wp_postmeta['post_id'].isin(list(wp_posts_post_type['ID'])))
                                            & (wp_postmeta['meta_key'].str[0] != '_')]
        all_tables_df[wp_type_posttype] = wp_postmeta_post_type.pivot(index='post_id', columns='meta_key',
                                                                      values='meta_value').reset_index()
        all_tables_df[wp_type_posttype] = all_tables_df[wp_type_posttype].merge(wp_posts_post_type, left_on='post_id',
                                                                                right_on='ID').drop(
            columns=['ID', 'post_type'])

    clean_wp_type_tables(all_tables_df)


def clean_wp_type_tables(all_tables_df):
    for table, column in [('wp_type_quote', 'bids'), ('wp_type_quote', 'chosen_bids'),
                          ('wp_type_agency', 'engineers'), ('wp_type_manufacturer', 'certifications')
                          ]:
        all_tables_df[table][column] = all_tables_df[table][column].apply(digit_array_of_digits_transform)
    clean_wp_type_quote(all_tables_df)
    clean_wp_type_part(all_tables_df)
    clean_wp_type_bid(all_tables_df)
    clean_wp_type_manufacturer(all_tables_df)
    clean_wp_type_agency(all_tables_df)


def clean_wp_type_agency(all_tables_df):
    # fill all_tables_df['wp_type_agency'] Nones with empty string in all columns
    all_tables_df['wp_type_agency'] = all_tables_df['wp_type_agency'].fillna('').astype('str')


def clean_wp_type_manufacturer(all_tables_df):
    all_tables_df['wp_type_manufacturer']['cnc_turning_notes'] = all_tables_df['wp_type_manufacturer']['cnc_turning_notes'].fillna('').astype('str')


def clean_wp_type_bid(all_tables_df):
    all_tables_df['wp_type_bid']['manufacturer'] = all_tables_df['wp_type_bid']['manufacturer'].astype('int64')


def clean_wp_type_part(all_tables_df):
    all_tables_df['wp_type_part']['unit_price'] = all_tables_df['wp_type_part']['unit_price'].fillna(-1).replace('', -1).astype('float')
    # Replace Null or empty strinc coc values with 'None'
    all_tables_df['wp_type_part']['coc'] = all_tables_df['wp_type_part']['coc'].fillna('None').replace('', 'None').astype('str')
    all_tables_df['wp_type_part']['quantity'] = all_tables_df['wp_type_part']['quantity'].fillna(-1).astype('int')


def clean_wp_type_quote(all_tables_df):
    logging.info("Cleaning wp_type_quote")
    df = all_tables_df['wp_type_quote']
    # Remove Avsha's test-agency (216)
    # Remove Ben's test-agency (439)
    df = df.drop(df[df['agency'].isin(["216", "439"])].index)
    df['bids'] = df['bids'].apply(get_bids_from_row)
    df['chosen_bids'] = df['chosen_bids'].apply(get_bids_from_row)
    all_tables_df['wp_type_quote'] = df
    return all_tables_df


def get_bids_from_row(bids_from_row) -> list:
    if bids_from_row is None:
        return []
    elif isinstance(bids_from_row, list):
        return bids_from_row
    elif bids_from_row.isdigit():
        return [int(bids_from_row)]
    elif isinstance(bids_from_row, str):
        if len(bids_from_row.strip()) == 0:
            return []
        bids_split = bids_from_row[1:-1].split(",")
        return [int(bid) for bid in bids_split]
    logging.error("Should not ever reach here, parsing error in some table, column")


def digit_array_of_digits_transform(digit_or_string):
    if (digit_or_string is None) or \
            (isinstance(digit_or_string, float) and math.isnan(digit_or_string)) or \
            (isinstance(digit_or_string, str) and len(digit_or_string) == 0):
        return []
    elif digit_or_string.isdigit():
        return digit_or_string
    elif isinstance(digit_or_string, str):
        return [int(x) for x in dict_to_list(loads(str.encode(digit_or_string)))]
    logging.error("Should not ever reach nere, parsing error in some table, column")


def clean_wp_parts(all_tables_df):
    df = all_tables_df['wp_parts']

    # Replace empty strings with NaN values in the 'quantity' column and then fill with zeros
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

    # Expand machining process
    df = pd.concat([df, pd.get_dummies(df['machining_process'])], axis=1)

    all_tables_df['wp_parts'] = df
    return all_tables_df


def clean_wp_manufacturers(all_tables_df):
    # remove test manufacturers
    test_manufacturers = [437, 590, 708, 1268, 24840]
    df = all_tables_df['wp_manufacturers']
    df = df.drop(df[df['post_id'].isin(test_manufacturers)].index)
    all_tables_df['wp_manufacturers'] = df
    # replace Empty strings in 'vendors' with 0
    all_tables_df['wp_manufacturers']['vendors'].replace('', 0, inplace=True)
    # replace Nons with 0:
    for nan_col in ['conventional_milling', 'conventional_turning', 'sheet_metal_press_break', 'sheet_metal_punching',
                    'sheet_metal_weldings', 'preffered_type_full_turnkey', 'preffered_type_assemblies']:
        all_tables_df['wp_manufacturers'][nan_col].fillna('0', inplace=True)
    # translate vendors type to int
    all_tables_df['wp_manufacturers']['vendors'] = all_tables_df['wp_manufacturers']['vendors'].astype('int64')
    all_tables_df['wp_manufacturers']['cnc_milling'] = all_tables_df['wp_manufacturers']['cnc_milling'].fillna(0).astype('int64')
    all_tables_df['wp_manufacturers']['cnc_turning'] = all_tables_df['wp_manufacturers']['cnc_turning'].fillna(0).astype('int64')
    # set 'house' column to type str and replace NaN with empty string '' (for later use in concat)
    all_tables_df['wp_manufacturers']['house'] = all_tables_df['wp_manufacturers']['house'].fillna('').astype('str')
    all_tables_df['wp_manufacturers']['cnc_turning_notes'] = all_tables_df['wp_manufacturers']['cnc_turning_notes'].fillna('').astype('str')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
