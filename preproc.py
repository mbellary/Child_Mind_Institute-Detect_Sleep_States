# Credit:
# This work is based on the top Kernel from the competition : https://www.kaggle.com/code/kmat2019/cmisleep-training-sample-2ndplace-kmat/input


import polars as pl
import numpy as np
import json
import math
import datetime
import argparse


class Config:
    def __init__(self, params_path):
        json_data = json.load(open(params_path))
        for k, v in json_data.items():
            setattr(self, k, v)

params_path = "params.json"
cfg = Config(params_path)

def set_table_dtypes(df):
    for col in df.columns:
        if col == 'event_start':
            df = df.with_columns(pl.col('event_start').cast(pl.Datetime))
    return df


def compute_offset_step(step: str) -> pl.Expr:
    start_timestamp = (pl.col('event_start').first())
    offset_step = (
        start_timestamp.dt.hour().cast(pl.Int64) * 60 * 12 +
        start_timestamp.dt.minute().cast(pl.Int64) * 12 +
        start_timestamp.dt.second().cast(pl.Int64) / 5
    )
    if step == 'daily':
        step = (pl.col('step') + offset_step) % (cfg.step_for_a_day)
    elif step == 'dayofweek':
        day_offset = (pl.col('step') + offset_step) // cfg.step_for_a_day
        step = start_timestamp.dt.weekday().cast(pl.Int64) + (day_offset  % 7) - 1
    return step

def timestamp_to_step(df) -> pl.Expr:
    step_df = (df.group_by('series_id')
                        .agg(
                            compute_offset_step('daily').cast(pl.Int64).alias('daily_step'),
                            compute_offset_step('dayofweek').alias('dayofweek')
                        )
                        .explode(['daily_step', 'dayofweek'])
                        .filter(pl.col('series_id') == '038441c925bb')
                        .with_row_index())
    df = df.join(step_df, how='inner', on='index').drop(pl.col('series_id_right'))
    return df

def compute_step():
    offset = (pl.col('event_end') - pl.col('event_start'))
    period = (
        offset.dt.total_hours() * 60 * 12
    )
    return period

def update_train_events(df) -> pl.Expr:
    df = df.with_columns(
            event_end = pl.when(pl.col('event_end').is_null()).then(pl.col('event_start').dt.offset_by('2h')).otherwise(pl.col('event_end')),
            event_start = pl.when(pl.col('event_start').is_null()).then(pl.col('event_end').dt.offset_by('-2h')).otherwise(pl.col('event_start')),
            target = pl.when(pl.col('event') == 'onset').then(0).otherwise(1)
         ).with_columns(
            target = pl.when(pl.col('event_start').is_null() & pl.col('event_end').is_null()).then(-1).otherwise(pl.col('target')),
            offset = compute_step()
         ).with_columns(
            pl.col('step', 'offset').shift(1).name.suffix("_onset_shiftdown"),
            pl.col('step').shift(-1).name.suffix("_wakeup_shiftup")
        ).with_columns(
            ~(pl.col('event_start').is_null() & pl.col('event_end').is_null() & pl.col('offset_onset_shiftdown').is_null()).alias('null_event')
        ).filter(
            pl.col('null_event')
        ).with_columns(
            step2 = pl.when(pl.col('step').is_null() & (pl.col('event') == 'onset'))
              .then((pl.col('step_onset_shiftdown')) + (pl.col('offset_onset_shiftdown')))
              .otherwise(pl.when(pl.col('step').is_null() & (pl.col('event') == 'wakeup'))
              .then((pl.col('step_wakeup_shiftup')) - (pl.col('offset')))
              .otherwise(pl.col('step')))
        ).drop(
            'step', 'offset', 'step_onset_shiftdown', 'offset_onset_shiftdown', 'step_wakeup_shiftup', 'null_event'
        ).with_columns(
            target_sw = 11
        )
    return df

sw_step_range = [1, 12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
sw_label = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

def compute_target_sw_before():
    for label, step_range in zip(sw_label, sw_step_range):
        yield (
            pl.when((pl.col('target_sw') == 11).shift(-step_range))
            .then(label)
            .otherwise(0).alias(f'sw_before_{label}')
            )

def compute_target_sw_after():
    for label, step_range in zip(sw_label, sw_step_range):
        yield (
            pl.when((pl.col('target_sw') == 11).shift(step_range))
            .then(label)
            .otherwise(0).alias(f'sw_after_{label}')
            )
    
def compute_forward_fill():
    for idx, (label, step_range) in enumerate(zip(sw_label, sw_step_range), start=0):
        offset = 0 if idx == 0 else sw_step_range[idx - 1]
        yield (pl.col(f'sw_before_{label}').forward_fill(step_range - offset - 1))

def compute_backward_fill():
    for idx, (label, step_range) in enumerate(zip(sw_label, sw_step_range), start=0):
        offset = 0 if idx == 0 else sw_step_range[idx - 1]
        yield (pl.col(f'sw_after_{label}').backward_fill(step_range - offset - 1))

def update_series_sw(df)-> pl.Expr:
    df = df.with_columns(
                compute_target_sw_before()
            ).with_columns(
                compute_target_sw_after()
            ).with_columns(
                pl.col("^sw_before_.*$" , "^sw_after_.*$").replace(0, None)
            ).with_columns(
                compute_forward_fill()
            ).with_columns(
                compute_backward_fill()
            ).with_columns(
                target_sw2 = pl.sum_horizontal("^sw_.*$", "target_sw")
            ).drop(
                "^sw.*$", "target_sw"
            ).rename(
                {"target_sw2" : "target_sw"}
            ).with_columns(
                pl.col("target_sw").replace(0, None)
            )
    return df

# Ensure that the data is re-indexed when padding for no data.
def pad(data):
    data = data.collect()
    left = data.select(pl.col('daily_step').first()).item()
    right_start_idx = data.select(pl.col('daily_step').last()).item()
    right = 17280 - 1 - (int(right_start_idx) % 17280)
    left_pad = data.clear(n=left)#.fill_null(0)
    right_pad = data.clear(n=right)#.fill_null(0)
    left_pad = left_pad.with_columns(daily_step = pl.int_range(0, left, 1))
    right_pad = right_pad.with_columns(daily_step = pl.int_range(right_start_idx + 1, 17280, 1))
    data_left = left_pad.vstack(data)
    series_data = data_left.vstack(right_pad)
    series_data = series_data.drop('index').with_row_index()
    return series_data.lazy()

def daily_counter(df)-> pl.Expr:
    daily_counter_df = (df.group_by('daily_step')
                         .agg(pl.col('index'), daily_counter = pl.col('mask').cum_sum())
                         .explode('index', 'daily_counter'))
    return (df.join(daily_counter_df, on='index', how='inner')
            .sort('index')
            .drop('daily_step_right')
            .fill_null(0))
    
def fill_null_date(df) -> pl.Expr:
    df = df.with_columns(
            pl.col('day').forward_fill(),
            pl.col('month').forward_fill(),
            pl.col('year').forward_fill()
           ).with_columns(
            pl.col('day').backward_fill(),
            pl.col('month').backward_fill(),
            pl.col('year').backward_fill()
           )
    return df

def get_time_mask_labels(df)-> pl.Expr:
    df = df.with_columns(
        time_label = "ts_" + pl.col('year').cast(pl.String) + pl.col("month").cast(pl.String) + pl.col("day").cast(pl.String),
        mask_label = "m_" + pl.col('year').cast(pl.String) + pl.col("month").cast(pl.String) + pl.col("day").cast(pl.String),
        delta_label = "dt_" + pl.col('year').cast(pl.String) + pl.col("month").cast(pl.String) + pl.col("day").cast(pl.String)
    )
    return df

def get_cols_expand(df)-> pl.Expr:
    time_unique_cols = sorted(np.array(df.select(pl.col("time_label").unique()).collect().to_series().to_list()).flatten().tolist())
    mask_unique_cols = sorted(np.array(df.select(pl.col("mask_label").unique()).collect().to_series().to_list()).flatten().tolist())
    delta_unique_cols = sorted(np.array(df.select(pl.col("delta_label").unique()).collect().to_series().to_list()).flatten().tolist())
    return time_unique_cols, mask_unique_cols, delta_unique_cols

# reshape from dayxfeatures to features x days
def add_day_mask(df)-> pl.Expr:
    agg_func = lambda col: col
    df_clone = df
    time_unique_cols,  mask_unique_cols, _ = get_cols_expand(df_clone)
    unique_cols = time_unique_cols + mask_unique_cols
    df_time = df.group_by(pl.col("month", "day")).agg(
                                (agg_func((pl.col("anglez").filter(pl.col("time_label") == value)).alias(value) for value in time_unique_cols)),
                             ).sort("month", "day").with_row_index()
    df_mask = df.group_by(pl.col("month", "day")).agg(
                                (agg_func((pl.col("mask").filter(pl.col("mask_label") == value)).alias(value) for value in mask_unique_cols)),
                             ).sort("month", "day").with_row_index()
    df_day_cols = df_time.join(df_mask, on="index").drop("month_right", "day_right", "index")
    for col in unique_cols:
        val = df_day_cols.select(pl.col(col).filter(pl.col(col).list.first().is_not_null())).collect().to_series().to_list()[0]
        df_day_cols = df_day_cols.with_columns(
                            pl.when(pl.col(col).list.first().is_not_null())
                              .then(pl.col(col))
                              .otherwise(val))
    df_day_cols = df_day_cols.explode(unique_cols).sort("month", "day").with_row_index()
    df = df.join(df_day_cols, on="index")
    return df

def reshape_delta_matrix(df)-> pl.Expr:
    agg_func = lambda col: col
    _,  _, delta_unique_cols = get_cols_expand(df)
    unique_cols = delta_unique_cols
    df_delta = df.group_by(pl.col("month", "day")).agg(
                                (agg_func((pl.col("delta_matrix").filter(pl.col("delta_label") == value)).alias(value) for value in delta_unique_cols)),
                             ).sort("month", "day")
    for col in unique_cols:
        val = df_delta.select(pl.col(col).filter(pl.col(col).list.first().is_not_null())).collect().to_series().to_list()[0]
        df_delta = df_delta.with_columns(
                            pl.when(pl.col(col).list.first().is_not_null())
                              .then(pl.col(col))
                              .otherwise(val))
    df_delta = df_delta.explode(unique_cols).sort("month", "day").with_row_index()
    # df = df.join(df_delta, on="index")
    return df_delta

def unnest_list_structs(df)-> pl.Expr:
    unique_cumsum = pl.DataFrame({"nan_counter_fold" : df.select(pl.col("nan_counter_fold").first()).collect().to_series()[0]}).unnest("nan_counter_fold")
    df_cumsum_unnested = pl.concat(unique_cumsum.select(pl.all())).rename("nan_counter")
    return df_cumsum_unnested

def get_nan_counter(df)-> pl.Expr:
    df_delta = reshape_delta_matrix(df)
    df_delta = df_delta.with_columns(
                            pl.cum_fold( acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=(pl.col("^dt_.*$") > 0 )).alias("nan_counter_fold")
                    ).group_by("month", "day").agg(pl.col("nan_counter_fold")).sort("month", "day")
    df_cumsum_unnested = pl.DataFrame(unnest_list_structs(df_delta)).with_row_index().lazy()
    df_nan_counter = df.join(df_cumsum_unnested, on="index")
    return df_nan_counter

def compute_delta_mask_matrix(df)-> pl.Expr:
    time_unique_cols, mask_unique_cols, _ = get_cols_expand(df)
    for time_col, mask_col in zip(time_unique_cols, mask_unique_cols):
        df = df.with_columns(
                (pl.col(time_col) - pl.col("anglez")).alias(f"diff_time_{time_col}"),
                (pl.col(mask_col) * pl.col("mask")).alias(f"diff_mask_{mask_col}")
        ).with_columns(
            ((pl.col(f"diff_time_{time_col}") == 0) * pl.col(f"diff_mask_{mask_col}")).alias(f"delta_matrix_full_{time_col}")
        )
    df = df.with_columns(
                (pl.sum_horizontal("^delta_matrix_full_.*$") - 1).alias(f"delta_matrix"),
                pl.sum_horizontal("^diff_mask.*$").alias(f"mask_matrix")
    ).drop("^ts_.*$", "^m_.*$", "^diff_time.*$", "^diff_mask.*$", "month_right", "day_right", "^delta_matrix_full_.*$")
    return df

def get_anglez_sum(df)-> pl.Expr:
    df_grouped = df.group_by("daily_step").agg(pl.col("anglez_valid").sum().alias("anglez_sum"))
    df = df.join(df_grouped, on="daily_step")
    return df

def get_valid_sum(df)-> pl.Expr:
    df_grouped = df.group_by("daily_step").agg(pl.col("valid").sum().alias("valid_sum"))
    df = df.join(df_grouped, on="daily_step")
    return df

def get_anglez_dev(df)-> pl.Expr:
    df_anglez_dev = df.group_by("daily_step").agg((pl.col("anglez_dev")**2).sum().alias("anglez_dev_sqr"))
    df = df.join(df_anglez_dev, on="daily_step")
    return df
    
def anglez_enmo_to_az(df)-> pl.Expr:
    df= df.with_columns(
            axay = (((pl.col("enmo") + 1) ** 2) / ( 1 + ((((pl.col("anglez") / 180) * math.pi).clip(-0.99999, 0.99999)) ** 2))).sqrt()
        ).with_columns(
            az = ((pl.col("anglez") / 180) * math.pi).clip(-0.99999, 0.99999) * pl.col("axay")
        )
    return df

def preprocess_features(df)-> pl.Expr:
    df = df.with_columns(
            daily_step = pl.col("daily_step") / cfg.step_for_a_day,
            anglez = pl.col("anglez") / 90,
            anglez_nanexist = pl.col("anglez_nanexist").cast(pl.Float32),
            anglez_daystd = (pl.col("anglez_daystd") / 90).clip(0, 2),
            anglez_daymean = (pl.col("anglez_daymean") / 90).clip(0, 2),
            anglez_daycounter = pl.col("anglez_daycounter") / (pl.col("anglez_daycounter").max()),
            anglez_nancounter = (pl.col("anglez_nancounter") > 2).cast(pl.Float32),
            enmo = pl.col("enmo").log1p().clip(0, 5),
            step_count = pl.col("step") / (pl.col("step").max()),
            dayofweek = pl.col("dayofweek") / 7.,
            target = pl.col("target").cast(pl.Float32),
            target_sw = pl.col("target_sw").cast(pl.Float32)
            )
    return df


def main():

	train_events = (pl.scan_csv(cfg.train_events_path)
						.rename({"timestamp": "event_start"})
						.filter(pl.col('series_id') == '038441c925bb'))
	train_series = (pl.scan_parquet(cfg.train_series_path)
						.with_row_index()
						.rename({"timestamp": "event_start"})
						.filter(pl.col('series_id') == '038441c925bb' ))#.with_row_index()
	train_events = (train_events
	            .pipe(set_table_dtypes)
	            .with_columns(pl.col('event_start').dt.offset_by("-5s").shift(-1).alias('event_end'))
	            .pipe(update_train_events)
	           )
	train_series = (train_series
	            .pipe(set_table_dtypes)
	            .pipe(timestamp_to_step)
	            .join(  train_events, left_on='step', right_on='step2', how='left')
	            .drop(["series_id_right" , "night", "event", "event_start_right"])
	            .with_columns(pl.col('target').forward_fill().fill_null(-1))
	            .pipe(update_series_sw)
	            .with_columns(diff_anglez = abs(pl.col('anglez').diff().fill_null(0)), mask = pl.when(pl.col('anglez').is_not_null()).then(1).otherwise(0))
	            .pipe(pad)
	            .pipe(daily_counter)
	            .with_columns(pl.col('event_start').dt.day().alias('day'), pl.col('event_start').dt.month().alias('month'), pl.col('event_start').dt.year().alias('year'))
	            .pipe(fill_null_date)
	            .pipe(get_time_mask_labels)
	            .filter(pl.col("day").is_in([14, 15, 16])) # <-----
	            .pipe(add_day_mask)
	            .pipe(compute_delta_mask_matrix)
	            .pipe(get_nan_counter)
	            .with_columns(
	                (pl.col("delta_matrix") > 0).alias("nan_exist_other_day"),
	                (pl.col("delta_matrix") == 0).alias("maybe_not_nan")
	           ).with_columns(
	                (pl.col("mask") * pl.col("maybe_not_nan")).alias("valid"),
	                (pl.col("diff_anglez") * pl.col("maybe_not_nan")).alias("anglez_valid")
	           ).pipe(get_anglez_sum)
	            .pipe(get_valid_sum)
	            .with_columns(anglez_mean = pl.col("anglez_sum") / pl.col("valid_sum"))
	            .with_columns(anglez_dev = (pl.col("anglez_valid") - pl.col("anglez_mean")) * pl.col("valid"))
	            .pipe(get_anglez_dev)
	            .with_columns(anglez_std = (pl.col("anglez_dev_sqr") / pl.col("valid_sum")).sqrt())
	            .rename(
	                {
	                    "delta_matrix" : "anglez_numrepeat",
	                    "mask_matrix" : "anglez_daycounter",
	                    "nan_counter" : "anglez_nancounter",
	                    "nan_exist_other_day" : "anglez_nanexist",
	                    "anglez_mean" : "anglez_daymean",
	                    "anglez_std" : "anglez_daystd",
	                    "daily_counter" : "anglez_counter"
	                }
	            ).pipe(anglez_enmo_to_az)
	            .pipe(preprocess_features)
	        )
	print(train_series.collect().head(5))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Preprocess Child Minde Institute Detect Sleep states training data')
	args = parser.parse_args()
	main()