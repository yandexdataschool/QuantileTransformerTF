import os
import tensorflow as tf
import numpy as np
import scipy.stats
from sklearn.preprocessing import QuantileTransformer
from quantile_transformer_tf import QuantileTransformerTF
from quantile_transformer_tf.interpolate_tf import interp

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_transform():
    N = 10000
    rng = np.random.RandomState(223532)
    data = np.stack([
        rng.uniform(-10, 10, N),
        rng.lognormal(10, 5, N),
        rng.normal(0, 1, N),
        rng.normal(-1, 1, N)], axis=1)
    transformer = QuantileTransformer(
        output_distribution="normal",
        random_state=34214)
    data_transformed_sk = transformer.fit_transform(data)
    data_double_transformed_sk = transformer.inverse_transform(
        data_transformed_sk)
    np.testing.assert_allclose(data, data_double_transformed_sk)

    # To test that QuantileTransformerTF picks up the right columns
    # we ask it only for [1, 2, 3] columns and when testing use data[:, 1:]
    transformer_tf = QuantileTransformerTF(
        transformer, [1, 2, 3], dtype=np.float64)
    data_transformed_tf = transformer_tf.transform(
        data[:, 1:].astype(np.float64), False)
    data_double_transformed_tf = transformer_tf.transform(
        data_transformed_tf, True)

    with tf.Session() as session:
        data_transformed_tf_val, data_double_transformed_tf_val = session.run([
            data_transformed_tf, data_double_transformed_tf])
    np.testing.assert_allclose(data_transformed_sk[:, 1:], data_transformed_tf_val)
    np.testing.assert_allclose(data[:, 1:], data_double_transformed_tf_val)


def test_transform_default_params():
    N = 1000
    rng = np.random.RandomState(22922)
    data = np.stack([
        rng.lognormal(10, 5, N),
        rng.uniform(-10, 0, N),
        rng.normal(10, 10, N),
        rng.normal(-1, 1, N)], axis=1)
    transformer = QuantileTransformer(
        output_distribution="normal",
        random_state=3434)
    data_transformed_sk = transformer.fit_transform(data)
    data_double_transformed_sk = transformer.inverse_transform(
        data_transformed_sk)
    np.testing.assert_allclose(data, data_double_transformed_sk)

    transformer_tf = QuantileTransformerTF(transformer)
    data_transformed_tf = transformer_tf.transform(data.astype(np.float64), False)
    data_double_transformed_tf = transformer_tf.transform(
        data_transformed_tf, True)

    with tf.Session() as session:
        data_transformed_tf_val, data_double_transformed_tf_val = session.run([
            data_transformed_tf, data_double_transformed_tf])
    np.testing.assert_allclose(data_transformed_sk, data_transformed_tf_val)
    np.testing.assert_allclose(data, data_double_transformed_tf_val)


def test_interp():
    N = 10000
    train_x = np.linspace(-10, 10, num=N)
    train_y = scipy.stats.norm.pdf(train_x)
    test_x = np.array([-20, 0., 1., 2., 7.5, 20, 10, -10], dtype=np.float64)
    test_y_tf = interp(test_x, train_x, train_y)
    with tf.Session() as session:
        test_y_val = session.run(test_y_tf)
    interp_np = np.interp(test_x, train_x, train_y)
    np.testing.assert_allclose(interp_np, test_y_val)
