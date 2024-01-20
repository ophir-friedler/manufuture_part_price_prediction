# Dependency: all_tables_df['wp_type_bid']
def monthly_bid_success_rate_df(all_tables_df):
    # all_tables_df['monthly_bid_success_rate'] - monthly_bid_success_rate: Success rate dataframe
    bids_agg = all_tables_df['wp_type_bid'] \
        .groupby(['post_date_Ym', 'manufacturer', 'is_chosen'])[['post_id']] \
        .count() \
        .pivot_table('post_id', ['post_date_Ym', 'manufacturer'], 'is_chosen') \
        .fillna(0)
    bids_agg['success_rate'] = 100 * bids_agg[1] / (bids_agg[1] + bids_agg[0])
    bids_agg['total_bids'] = bids_agg[1] + bids_agg[0]
    all_tables_df['monthly_bid_success_rate'] = bids_agg


# Create: all_tables_df['monthly_projects_stats']
# columns: 'num_projects', 'num_projects_with_quote', 'num_projects_approved', 'pct_projects_with_quote',
#          'pct_approved_out_of_with_quote'
def monthly_projects_stats(all_tables_df):
    df = all_tables_df['wp_projects']

    # Project monthly aggregations
    df = df.groupby(['project_creation_date_Ym'])[
        ['project_creation_date_Ym', 'post_id', 'is_quote_carried_out', 'approval_date']].agg(
        {'post_id': ['count'],
         'is_quote_carried_out': ['sum'],
         'approval_date': ['count']}
    )
    df.columns = ['num_projects', 'num_projects_with_quote', 'num_projects_approved']
    df['pct_projects_with_quote'] = 100 * df['num_projects_with_quote'] / df['num_projects']
    df['pct_approved_out_of_with_quote'] = 100 * df['num_projects_approved'] / df['num_projects_with_quote']
    all_tables_df['monthly_projects_stats'] = df


# Dependencies: Enriched wp_manufacturers
def monthly_manufacturers_stats(all_tables_df):
    df = all_tables_df['wp_manufacturers']
    df = df.groupby(['manufacturer_creation_date_Ym'])[['post_id', 'manufacturer_creation_date_Ym']].agg(
        {'post_id': ['count']}
    )
    df.columns = ['num_manufacturers']
    all_tables_df['monthly_manufacturers_stats'] = df


# Dependencies: Enriched wp_type_quote
def stats_by_num_candidates(all_tables_df):
    df = all_tables_df['wp_type_quote'].groupby(['num_candidates', 'is_bid_chosen'])[['post_id']] \
        .count().pivot_table('post_id', ['num_candidates'], 'is_bid_chosen') \
        .reset_index().set_index('num_candidates') \
        .rename(columns={True: 'num_quotes_with_chosen_bid', False: 'num_quotes_without_chosen_bid'})
    # Calculate % of success per number of candidates
    df['success_rate'] = 100.0 * df['num_quotes_with_chosen_bid'] / (df['num_quotes_with_chosen_bid'] + df['num_quotes_without_chosen_bid'])
    all_tables_df['stats_by_num_candidates'] = df
