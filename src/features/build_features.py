import math


def transform_to_comma_separated_str_set(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, list) and len(x) == 0):
        return None
    ret_val = ", ".join([str(y) for y in set([y for y in list(x) if y is not None and not (isinstance(y, float) and math.isnan(y))])])
    return "[" + ret_val + "]"


def get_first_material_category_level_1_set(werk_data_for_name_df):
    if len(werk_data_for_name_df) == 0:
        return None
    return werk_data_for_name_df['material_category_level_1_set'].iloc[0]
