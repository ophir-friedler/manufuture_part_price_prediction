import math


def transform_to_comma_separated_str_set(x):
    # if x is None or x is Nan or x is empty list, return None
    if x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, list) and len(x) == 0):
        return None
    # For each value in x, if it is not None or Nan, add it to the set, then return the set as a comma separated string, make sure you return a set and not a list
    ret_val = ", ".join([str(y) for y in set([y for y in list(x) if y is not None and not (isinstance(y, float) and math.isnan(y))])])
    return "[" + ret_val + "]"
