import tensorflow as tf


def nonzero(tensor):
    return tf.cast(tf.where(tensor)[:, 0], tf.int32)


def searchsorted(sorted_array, values_to_insert, scope=None):
    """
    Determine indices within a sorted array at which to insert elements.
    Equivalent of `np.searchsorted`.
    Currently only 1D arrays are supported, but should be easy to extend.
    https://gist.github.com/joshburkart/60de56294b472b75206c3f29c2c81375
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


class InterpolatorTF():
    """
    A 1D linear interpolator
    """

    def fit(self, train_x, train_y):
        """
        Fits the interpolator. Makes sense to explicitely compute the
        line coefficients only once.
        Args:
        train_x: tf.Tensor[n_examples], x-coordinates of the training data
            Must be in increasing order. This is not checked.
        train_y: tf.Tensor[n_examples], y-coordinates of the training data
        """
        self.line_a = train_y[:-1]
        self.line_b = (-train_y[:-1] + train_y[1:])/(-train_x[:-1] + train_x[1:])
        self.references = train_x
        self.high_x = train_x[-1]
        self.low_x = train_x[0]
        self.low_y = train_y[0]
        self.high_y = train_y[-1]
        self.y_dtype = train_y.dtype
        return self

    def interp(self, query_x):
        """
        Computes the interpolation at the given points. Uses the
         first and last traing values for queries outside of the
         training range.
        Args:
        query_x: tf.Tensor[n_query_points], x-coordinates where to evaluate the interpolation
        Returns:
        tf.Tensor[n_query_points], interpolated y-coordinates
        """
        overflow_mask = (query_x >= self.high_x)
        underflow_mask = (query_x <= self.low_x)
        in_range_mask = tf.logical_not(tf.logical_or(overflow_mask, underflow_mask))
        return tf.dynamic_stitch(
            [
                nonzero(overflow_mask),
                nonzero(underflow_mask),
                nonzero(in_range_mask)],
            [
                self.high_y*tf.ones(tf.count_nonzero(overflow_mask), dtype=self.y_dtype),
                self.low_y*tf.ones(tf.count_nonzero(underflow_mask), dtype=self.y_dtype),
                self._interp_inner(tf.boolean_mask(query_x, in_range_mask))
            ])

    def _interp_inner(self, query_x):
        query_indices = searchsorted(self.references, query_x) - 1
        return tf.gather(self.line_a, query_indices) + tf.gather(self.line_b, query_indices)*(
            query_x - tf.gather(self.references, query_indices))


def interp(x, xp, fp):
    """
    numpy-like interpolation function
    Args:
    x: tf.Tensor[n_query] the x-coordinates at which to evaluate the interpolated values.
    xp: tf.Tensor[n_train] the x-coordinates of the training data
            Must be in increasing order. This is not checked.
    fp: tf.Tensor[n_train], y-coordinates of the training data
    Returns:
    tf.Tensor[n_query_points], interpolated y-coordinates
    """
    interpolator = InterpolatorTF()
    interpolator.fit(xp, fp)
    return interpolator.interp(x)
