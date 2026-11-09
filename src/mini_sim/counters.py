import numpy as np

# The counters stored for each tile and added up across tiles.
# skipped_macs and effective_weight_density are worked out from these.
COUNTERS = [
    "dense_macs",
    "issued_macs",
    "zero_slot_macs",
    "useful_macs",
    "nonzero_weights",
    "weight_elements"]

def add_skipped_and_density(counters):
    counters["skipped_macs"] = counters["dense_macs"] - counters["issued_macs"]
    counters["effective_weight_density"] = counters["nonzero_weights"] / counters["weight_elements"]
    return counters

def dense_tile_counters(rows, w):
    num_k = w.shape[0]
    num_columns = w.shape[1]
    nonzero_weights = int(np.count_nonzero(w))
    total_weights = num_k * num_columns

    tile_counters = {}
    tile_counters["dense_macs"] = rows * total_weights
    tile_counters["issued_macs"] = rows * total_weights
    tile_counters["zero_slot_macs"] = rows * (total_weights - nonzero_weights)
    tile_counters["useful_macs"] = rows * nonzero_weights
    tile_counters["nonzero_weights"] = nonzero_weights
    tile_counters["weight_elements"] = total_weights

    return add_skipped_and_density(tile_counters)

def sparse_tile_counters(rows, values):
    # K / 2
    kept_per_column = values.shape[0]
    num_columns = values.shape[1]
    nonzero_weights = int(np.count_nonzero(values))
    kept_weights = kept_per_column * num_columns
    total_weights = 2 * kept_weights

    tile_counters = {}
    tile_counters["dense_macs"] = rows * total_weights
    tile_counters["issued_macs"] = rows * kept_weights
    tile_counters["zero_slot_macs"] = rows * (kept_weights - nonzero_weights)
    tile_counters["useful_macs"] = rows * nonzero_weights
    tile_counters["nonzero_weights"] = nonzero_weights
    tile_counters["weight_elements"] = total_weights

    return add_skipped_and_density(tile_counters)

def sum_counters(list_of_counters):
    totals = {}
    for name in COUNTERS:
        totals[name] = 0
    for tile in list_of_counters:
        for name in COUNTERS:
            totals[name] += int(tile[name])

    return add_skipped_and_density(totals)