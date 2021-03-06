import tensorflow as tf
import numpy as np
from collections import namedtuple

from .interpolate_tf import InterpolatorTF, nonzero

InterpolatorsTuple = namedtuple(
    "InterpolatorsTuple",
    [
        "quantiles_to_references_forward",
        "quantiles_to_references_backward",
        "references_to_quantiles",
        "low_quantile",
        "high_quantile"
    ])


class QuantileTransformerTF():
    """sklearn.preprocessing.QuantileTransformer that can be applied in Tensorflow

    From the sklean documentation:
    Transform features using quantiles information.

    This method transforms the features to follow a uniform or a
    normal distribution. Therefore, for a given feature, this
    transformation tends to spread out the most frequent values. It
    also reduces the impact of (marginal) outliers: this is therefore
    a robust preprocessing scheme. The transformation is applied on
    each feature independently. The cumulative density function of a
    feature is used to project the original values. Features values of
    new/unseen data that fall below or above the fitted range will be
    mapped to the bounds of the output distribution. Note that this
    transform is non-linear. It may distort linear correlations
    between variables measured at the same scale but renders variables
    measured at different scales more directly comparable.

    """

    scope = "QuantileTransformerTF"

    def in_tf_scope(function):
        def res(self, *args, **kwargs):
            with tf.name_scope(self.scope):
                return function(self, *args, **kwargs)
        return res

    @in_tf_scope
    def __init__(self, sklearn_transformer, sklearn_indices=None, dtype=None):
        """
        Args:
        sklearn_transformer: instance of fitted sklearn.preprocessing.QuantileTransformer
        sklearn_indices: list of feature indices to use. E. g. if you trained
           a transformer for features+outputs, here you can get separate ones. If
           None, takes all the features
        dtype: np.float32/np.float64, the dtype the transformer expects and outputs.
           If None defaults to the sklearn_transformer.quantiles_.dtype
        """
        if sklearn_transformer.output_distribution != 'normal':
            raise ValueError("Only normal distribution is supported")

        if dtype is None:
            dtype = sklearn_transformer.quantiles_.dtype.type

        self.output_distribution = tf.distributions.Normal(
            dtype(0), dtype(1), name="output_distribution")

        if sklearn_indices is not None:
            selected_quantiles = sklearn_transformer.quantiles_[:, sklearn_indices]
        else:
            selected_quantiles = sklearn_transformer.quantiles_

        self._quantiles = tf.constant(selected_quantiles.astype(dtype),
                                      name="quantiles")
        self._references = tf.constant(sklearn_transformer.references_.astype(dtype),
                                       name="references")
        self.n_colunms = selected_quantiles.shape[1]
        self.interpolators_by_index = []
        for index in range(self.n_colunms):
            interpolator_quantiles_to_references_forward = InterpolatorTF().fit(
                self._quantiles[:, index], self._references)
            interpolator_quantiles_to_references_backward = InterpolatorTF().fit(
                -self._quantiles[::-1, index], -self._references[::-1])
            interpolator_references_to_quantiles = InterpolatorTF().fit(
                self._references, self._quantiles[:, index])
            self.interpolators_by_index.append(InterpolatorsTuple(
                interpolator_quantiles_to_references_forward,
                interpolator_quantiles_to_references_backward,
                interpolator_references_to_quantiles,
                self._quantiles[0, index],
                self._quantiles[-1, index]))
        self.BOUNDS_THRESHOLD = dtype(1e-7)
        self.dtype = dtype

    @in_tf_scope
    def transform(self, data, inverse):
        """
        Builds a graph for transformation
        Args:
        data - tf.Tensor[n_examples, n_features]
        inverse - bool, whether inverse or forward transform is desired

        Returns:
        tf.Tensor[n_examples, n_features] - transformed data
        """
        if inverse:
            data = self.output_distribution.cdf(data)
        per_feature_transformed = []
        for i in range(self.n_colunms):
            this_transformed = self._transform_col(data[:, i],
                                                   self.interpolators_by_index[i],
                                                   inverse)
            this_transformed.set_shape([data.shape[0]])
            per_feature_transformed.append(this_transformed)
        return tf.stack(per_feature_transformed, axis=1)

    def inverse_transform(self, data):
        """
        Builds a graph for inverse transformation
        Args:
        data - tf.Tensor[n_examples, n_features]

        Returns:
        tf.Tensor[n_examples, n_features] - transformed data
        """
        return self.transform(data, inverse=True)

    @in_tf_scope
    def _transform_col(self, data, interpolators, inverse):
        if not inverse:
            lower_bound_x = interpolators.low_quantile
            upper_bound_x = interpolators.high_quantile
            lower_bound_y = self.dtype(0)
            upper_bound_y = self.dtype(1)
        else:
            lower_bound_x = self.dtype(0)
            upper_bound_x = self.dtype(1)
            lower_bound_y = interpolators.low_quantile
            upper_bound_y = interpolators.high_quantile

        lower_bounds_mask = (data - self.BOUNDS_THRESHOLD < lower_bound_x)
        upper_bounds_mask = (data + self.BOUNDS_THRESHOLD > upper_bound_x)
        in_range_mask = tf.logical_not(tf.logical_or(lower_bounds_mask, upper_bounds_mask))
        data_in_range = tf.boolean_mask(data, in_range_mask)

        if not inverse:
            interpolated = 0.5*(
                interpolators.quantiles_to_references_forward.interp(data_in_range) -
                interpolators.quantiles_to_references_backward.interp(-data_in_range))
        else:
            interpolated = interpolators.references_to_quantiles.interp(data_in_range)

        res = tf.dynamic_stitch(
            [nonzero(upper_bounds_mask),
             nonzero(in_range_mask),
             nonzero(lower_bounds_mask)],
            [tf.fill(tf.count_nonzero(upper_bounds_mask, keepdims=True), upper_bound_y),
             interpolated,
             tf.fill(tf.count_nonzero(lower_bounds_mask, keepdims=True), lower_bound_y)])

        if not inverse:
            res = self.output_distribution.quantile(res)
            clip_min = self.output_distribution.quantile(tf.constant(
                self.BOUNDS_THRESHOLD - np.spacing(1), dtype=self.dtype))
            clip_max = self.output_distribution.quantile(tf.constant(
                1 - (self.BOUNDS_THRESHOLD - np.spacing(1)), dtype=self.dtype))
            res = tf.clip_by_value(res, clip_min, clip_max)
        return res
