import tensorflow as tf
import numpy as np

def nonzero(tensor):
    return tf.cast(tf.where(tensor)[:,0], tf.int32)

# https://gist.github.com/joshburkart/60de56294b472b75206c3f29c2c81375
def searchsorted(sorted_array, values_to_insert, scope=None):
    """Determine indices within a sorted array at which to insert elements.
    Equivalent of `np.searchsorted`.
    Currently only 1D arrays are supported, but should be easy to extend.
    """
    with tf.name_scope(scope, default_name='searchsorted'):
        sorted_array_tens = tf.convert_to_tensor(sorted_array)
        values_tens = tf.convert_to_tensor(values_to_insert)

        sorted_array_shape = tf.shape(sorted_array_tens)
        values_shape = tf.shape(values_tens)

        sorted_array_len = sorted_array_shape[0]
        values_len = values_shape[0]

        # Put everything together and argsort.
        concat = tf.concat([sorted_array_tens, values_tens], axis=0)
        concat_argsort = tf.contrib.framework.argsort(concat)  # pylint: disable=no-member

        # Find indices into `sorted_array` where values should be inserted (but
        # not yet in the right order).
        insert_mask = concat_argsort >= sorted_array_len
        sorted_insert_indices = tf.reshape(
            tf.where(insert_mask), values_shape) - tf.range(
                tf.to_int64(values_len))

        # Construct permutation specifying ordering of insertion indices.
        insert_perm = tf.boolean_mask(concat_argsort,
                                      insert_mask) - sorted_array_len
        # The permutation we created satisfies, in NumPy notation,
        # `sorted_insert_indices = insert_indices[insert_perm]`. With TF this is
        # instead `sorted_insert_indices = tf.gather(insert_indices,
        # insert_perm)`. But we want `insert_indices`, so we need the inverse,
        # which is `tf.scatter_nd`.
        insert_indices = tf.scatter_nd(
            tf.to_int64(insert_perm)[..., tf.newaxis],
            sorted_insert_indices,
            shape=tf.to_int64(values_shape))
    return insert_indices


def interp_inner(query_x, train_x, train_y):
    line_a = train_y[:-1]
    line_b = (-train_y[:-1] + train_y[1:])/(- train_x[:-1] + train_x[1:])
    query_indices = searchsorted(train_x, query_x) - 1
    return tf.gather(line_a, query_indices) + tf.gather(line_b, query_indices)*(
        query_x - tf.gather(train_x, query_indices))


def interp(query_x, train_x, train_y):
    overflow_mask = (query_x >= train_x[-1])
    underflow_mask = (query_x <= train_x[0])
    in_range_mask = tf.logical_not(tf.logical_or(overflow_mask, underflow_mask))
    return tf.dynamic_stitch(
        [
            nonzero(overflow_mask),
            nonzero(underflow_mask),
            nonzero(in_range_mask)],
        [
            train_y[-1]*tf.ones(tf.count_nonzero(overflow_mask), dtype=train_y.dtype),
            train_y[0]*tf.ones(tf.count_nonzero(underflow_mask), dtype=train_y.dtype),
            interp_inner(
                tf.boolean_mask(query_x, in_range_mask),
                train_x,
                train_y)])


class QuantileTransformerTF():
    def __init__(self, sklearn_transformer, sklearn_indices, dtype):
        if sklearn_transformer.output_distribution != 'normal':
            raise ValueError("Only normal distribution is supported")

        with tf.name_scope("QuantileTransformerTF"):
            self.output_distribution = tf.distributions.Normal(
                dtype(0), dtype(1), name="output_distribution")
            self.quantiles = tf.constant(
                sklearn_transformer.quantiles_[:, sklearn_indices].astype(dtype),
                name="quantiles")
            self.references = tf.constant(sklearn_transformer.references_.astype(dtype),
                                          name="references")
            self.n_colunms = len(sklearn_indices)

        self.BOUNDS_THRESHOLD = dtype(1e-7)
        self.dtype = dtype

    def transform(self, data, inverse):
        with tf.name_scope("QuantileTransformerTF"):
            if inverse:
                data = self.output_distribution.cdf(data)
            per_feature_transformed = []
            for i in range(self.n_colunms):
                this_transformed = self._transform_col(data[:, i],
                                                       self.quantiles[:, i],
                                                       inverse)
                tf.assert_equal(tf.shape(this_transformed)[0], tf.shape(data)[0])
                this_transformed.set_shape([data.shape[0]])
                per_feature_transformed.append(this_transformed)
                
            return tf.stack(per_feature_transformed, axis=1)

    def _transform_col(self, data, quantiles, inverse):
        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = self.dtype(0)
            upper_bound_y = self.dtype(1)
        else:
            lower_bound_x = self.dtype(0)
            upper_bound_x = self.dtype(1)
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]

        lower_bounds_idx = (data - self.BOUNDS_THRESHOLD < lower_bound_x)
        upper_bounds_idx = (data + self.BOUNDS_THRESHOLD > upper_bound_x)

        if not inverse:
            forward = interp(data, quantiles, self.references)
            backward = interp(-data, -quantiles[::-1], -self.references[::-1])
            interpolated = 0.5*(forward - backward)
        else:
            interpolated = interp(data, self.references, quantiles)

        bounded = tf.dynamic_stitch(
            [nonzero(lower_bounds_idx), nonzero(~lower_bounds_idx)],
            [lower_bound_y*tf.ones(tf.count_nonzero(lower_bounds_idx), dtype=self.dtype),
             tf.boolean_mask(interpolated, ~lower_bounds_idx)])

        res = tf.dynamic_stitch(
            [nonzero(upper_bounds_idx), nonzero(~upper_bounds_idx)],
            [upper_bound_y*tf.ones(tf.count_nonzero(upper_bounds_idx), dtype=self.dtype), 
             tf.boolean_mask(bounded, ~upper_bounds_idx)])

        if not inverse:
            res = self.output_distribution.quantile(res)
            clip_min = self.output_distribution.quantile(tf.constant(
                self.BOUNDS_THRESHOLD - np.spacing(1), dtype=self.dtype))
            clip_max = self.output_distribution.quantile(tf.constant(
                1 - (self.BOUNDS_THRESHOLD - np.spacing(1)), dtype=self.dtype))
            res = tf.clip_by_value(res, clip_min, clip_max)
        return res
