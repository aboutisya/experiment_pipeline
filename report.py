import pandas as pd
import numpy as np
import abc
import config as cfg
from tqdm import tqdm
from itertools import product
from metric_builder import Metric, CalculateMetric
from stattests import TTestFromStats, MannWhitney, ProportionsZtest, calculate_statistics, calculate_linearization


class Report:
    def __init__(self, report):
        self.report = report


class BuildMetricReport:
    def __call__(self, calculated_metric, metric_items, mc_estimator=None) -> Report:
        mappings_estimator = {
            "t_test": TTestFromStats(),
            "mann_whitney": MannWhitney(),
            "prop_test": ProportionsZtest()
        }

        if mc_estimator:
            estimator = mappings_estimator[mc_estimator]
        else:
            estimator = mappings_estimator[metric_items.estimator]
            cfg.logger.info(f"{metric_items.name}")

        stats = calculate_statistics(calculated_metric, metric_items.type)
        criteria_res = estimator(stats)

        report_items = pd.DataFrame({
            "metric_name": metric_items.name,
            "mean_0": stats.mean_0,
            "mean_1": stats.mean_1,
            "var_0": stats.var_0,
            "var_1": stats.var_1,
            "delta": stats.mean_1 - stats.mean_0,
            "lift":  (stats.mean_1 - stats.mean_0) / stats.mean_0,
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)


def build_experiment_report(df, metric_config, monte_carlo_config=None):
    build_metric_report = BuildMetricReport()
    reports = []

    if monte_carlo_config:
        for metric_params in metric_config:
            metric_parsed = Metric(metric_params)

            lifts = monte_carlo_config.get('lifts', cfg.DEFAULT_LIFTS_VALUE)
            for mc_lift in np.arange(lifts.get('start', cfg.DEFAULT_START_VALUE), lifts.get('end', cfg.DEFAULT_END_VALUE),
                                          step=lifts.get('by', cfg.DEFAULT_STEP_VALUE)):

                for estimator in monte_carlo_config.get('estimators', cfg.DEFAULT_ESTIMATOR):

                    cfg.logger.info(f"Metric: {metric_parsed.name} - Lift: {mc_lift} - Estimator: {estimator}")
                    metric_reports = []
                    for _ in tqdm(range(1000)):
                        calculated_metric = CalculateMetric(metric_parsed)(df, mc_lift=mc_lift)
                        metric_report = build_metric_report(calculated_metric, metric_parsed, mc_estimator=estimator)
                        metric_reports.append(metric_report.report)
                    metric_reports = pd.concat(metric_reports)
                    report_item = pd.DataFrame({
                        "metric_name": metric_parsed.name,
                        "lift": mc_lift,
                        "estimator": estimator,
                        "TPR": sum(np.array(np.where(metric_reports['pvalue'] < 0.05)))/1000,
                        "agreement": sum(np.array(np.where(metric_reports['lift'] > 0)))/1000 # предполагается, что в конфиге с монте-карло всегда положительные лифты
                    })
                    reports.append(Report(report_item).report)


        else:
            for metric_params in metric_config:
                metric_parsed = Metric(metric_params)
                calculated_metric = CalculateMetric(metric_parsed)(df)
                metric_report = build_metric_report(calculated_metric, metric_parsed)
                reports.append(metric_report.report)
    return pd.concat(reports)

