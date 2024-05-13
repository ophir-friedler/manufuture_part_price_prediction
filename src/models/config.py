PART_FEATURES_TO_TRAIN_ON = [
    'max_enclosing_cuboid_volume_bucketed',
    'average_tolerance_01_bucketed',
    'average_tolerance_001_bucketed',
    'average_tolerance_0001_bucketed',
    'average_number_of_nominal_sizes_bucketed',
    'first_material_categorization_level_1_set',
    'first_material_categorization_level_2_set',
    'quantity_bucket',
    'machining_process'
]
LIST_OF_RELU_LAYER_WIDTHS = [128, 64, 32]
BATCH_SIZE = 30
EPOCHS = 70
# PART_FEATURES_DICT = {'max_enclosing_cuboid_volume_bucketed': '[16384-32768)',
#                       'average_tolerance_01_bucketed': '[0-1)',
#                       'average_tolerance_001_bucketed': '[0-1)',
#                       'average_tolerance_0001_bucketed': '[0-1)',
#                       'average_number_of_nominal_sizes_bucketed': '[16-32)',
#                       'first_material_categorization_level_1_set': '[NONFERROUS_ALLOY]',
#                       'first_material_categorization_level_2_set': '[PLATINUM]',
#                       'quantity_bucket': '[4-5)',
#                       'machining_process': 'milling'}

PART_FEATURES_DICT = {"max_enclosing_cuboid_volume_bucketed": "[16384-32)",
                      "average_tolerance_01_bucketed": "[0-1)",
                      "average_tolerance_001_bucketed": "[0-1)",
                      "average_tolerance_0001_bucketed": "[0-1)",
                      "average_number_of_nominal_sizes_bucketed": "[16-32)",
                      "first_material_categorization_level_1_set": "[NONFERROUS_ALLOY]",
                      "first_material_categorization_level_2_set": "[PLATINUM]",
                      "quantity_bucket": "[4-5)",
                      "machining_process": "milling"
                      }

PART_PRICE_LABEL_FEATURE = 'unit_price'
MAX_ENCLOSING_CUBOID_VOLUMNE_NUM_EXPONENTIAL_BUCKETS = 25
AVERAGE_TOLERANCE_01_BUCKETED_EXPONENTIAL_BUCKETS = 10
