from src.config import PART_PRICE_TRAINING_TABLE_NAME

SKIPPED_RAW_MANUFUTURE_TABLES = ['wp_actionscheduler_actions', 'wp_wc_order_stats']
PRICE_BUCKETS = [1, 1.5, 2.5, 4, 6, 10, 15, 20, 30, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400,
                 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
MANUFACTURER_BID_LABEL_COLUMN_NAME = 'is_manuf_bid'
MIN_NUM_BIDS_PER_MANUFACTURER = 4


COUNTRY_TO_ISO_MAP = {'IL': 'IL',
                      'CN': 'CN',
                      'HU': 'HU',
                      'US': 'US',
                      'TR': 'TR',
                      'TW': 'TW',
                      'KP': 'KP',
                      'MX': 'MX',
                      'AF': 'AF',
                      'IN': 'IN',
                      'SV': 'SV',
                      'HK': 'HK',
                      'KR': 'KR',
                      'IT': 'IT',
                      'TH': 'TH',
                      'VN': 'VN',
                      'India': 'IN',
                      'Israel': 'IL',
                      'China': 'CN',
                      'Afghanistan': 'AF',
                      'El Salvador': 'SV',
                      'Hong Kong': 'HK',
                      'GB': 'GB',
                      'RS': 'RS'
                      }

TABLES_TO_SAVE_TO_DB = [PART_PRICE_TRAINING_TABLE_NAME, 'wp_type_part']
