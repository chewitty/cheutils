import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from cheutils.common_utils import get_quantiles
from cheutils.loggers import LoguruWrapper
from scipy.stats import iqr

LOGGER = LoguruWrapper().get_logger()

class ExtremeStateAugmenter(BaseEstimator, TransformerMixin):
    def __init__(self, rel_cols: list, lower_quartiles: list, upper_quartiles: list, group_by: list=None, ):
        """
        Create a new ExtremeStateAugmenter instance.
        :param rel_cols: the list of columns with features to examine for extree values
        :param lower_quartiles: list of corresponding lower quartile (float between 0 and 1), matching the relevant feature columns specified
        :param upper_quartiles: list of corresponding upper quartile (float between 0 and 1 but higher than the lower quartiles), matching the relevant feature columns specified
        :param group_by: any necessary category to group aggregate stats by - default is None
        """
        assert rel_cols is not None or not (not rel_cols), 'Valid numeric feature columns must be specified'
        assert lower_quartiles is not None or not (not lower_quartiles), 'Valid lower quartiles for the numeric features must be specified'
        assert upper_quartiles is not None or not (not upper_quartiles), 'Valid upper quartiles for the numeric features must be specified'
        self.rel_cols = rel_cols
        self.lower_quartiles = lower_quartiles
        self.upper_quartiles = upper_quartiles
        self.group_by = group_by
        self.inter_quartile_ranges = {}
        self.global_inter_quartile_ranges = {}
        self.fitted = False

    def fit(self, X=None, y=None):
        if self.fitted:
            return self
        LOGGER.debug('ExtremeStateAugmenter: Fitting dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        if self.group_by is not None and not (not self.group_by):
            cur_groups = X.groupby(self.group_by)
            for grp_name, group in cur_groups:
                cur_iqrs = {}
                for idx, rel_col in enumerate(self.rel_cols):
                    col_iqr = iqr(group[rel_col])
                    qvals = get_quantiles(group, rel_col, [self.lower_quartiles[idx], self.upper_quartiles[idx]])
                    l_thres = qvals[0] - 1.5 * col_iqr
                    u_thres = qvals[1] + 1.5 * col_iqr
                    cur_iqrs[rel_col] = (l_thres, u_thres)
                self.inter_quartile_ranges[grp_name] = cur_iqrs
        else:
            for rel_col in self.rel_cols:
                col_iqr = iqr(X[rel_col])
                qvals = get_quantiles(X, rel_col, [self.lower_quartiles, self.upper_quartiles])
                l_thres = qvals[0] - 1.5 * col_iqr
                u_thres = qvals[1] + 1.5 * col_iqr
                self.global_inter_quartile_ranges[rel_col] = (l_thres, u_thres)
        self.fitted = True
        return self

    def transform(self, X, **fit_params):
        LOGGER.debug('ExtremeStateAugmenter: Transforming dataset, shape = {}, {}', X.shape, fit_params)
        new_X = self.__do_transform(X, y=None, **fit_params)
        LOGGER.debug('ExtremeStateAugmenter: Transformed dataset, shape = {}, {}', new_X.shape, fit_params)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        LOGGER.debug('ExtremeStateAugmenter: Fitting and transforming dataset, shape = {}, {}', X.shape, y.shape if y is not None else None)
        self.fit(X, y)
        new_X = self.__do_transform(X, y, **fit_params)
        LOGGER.debug('ExtremeStateAugmenter: Fit-transformed dataset, shape = {}, {}', new_X.shape, y.shape if y is not None else None)
        return new_X

    def __do_transform(self, X, y=None, **fit_params):
        if not self.fitted:
            raise RuntimeError('You have to call fit on the transformer before')
        # apply sine and cosine transformations appropriately
        new_X = X
        if self.group_by is not None and not (not self.group_by):
            cur_groups = X.groupby(self.group_by)
            grp_subset = []
            renamed_cols = [rel_col + '_extreme' for rel_col in self.rel_cols]
            for grp_name, group in cur_groups:
                grp_iqrs = self.inter_quartile_ranges[grp_name]
                for col_name, rel_col in zip(renamed_cols, self.rel_cols):
                    prevailing_iqrs = grp_iqrs.get(rel_col)
                    group.loc[:, col_name] = (group[rel_col] < prevailing_iqrs[0]) | (group[rel_col] > prevailing_iqrs[1])
                grp_subset.append(group[renamed_cols])
            updated_srs = pd.concat(grp_subset, ignore_index=False, )
            new_X.loc[:, renamed_cols] = updated_srs.astype(int)
        else:
            for rel_col in self.rel_cols:
                prevailing_iqrs = self.global_inter_quartile_ranges.get(rel_col)
                new_X.loc[:, rel_col + '_extreme'] = new_X[[rel_col]] < prevailing_iqrs[0] or new_X[[rel_col]] > prevailing_iqrs[1]
        return new_X