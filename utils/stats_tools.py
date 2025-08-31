import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from utils.data_pipeline import data_explicit_preprocess
from utils.model_tools import item_name

data_explicit_preprocess([
item_name("Bandos chestplate"),
item_name("Bandos tassets"),
item_name("Bandos boots"),
item_name("Bandos hilt"),
item_name("Godsword shard 1"),
item_name("Godsword shard 2"),
item_name("Godsword shard 3"),
])