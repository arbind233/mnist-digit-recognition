import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mnist_cnn.keras")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test[..., None] / 255.0

def test_output_shape():
    assert model.predict(x_test[:1], verbose=0).shape == (1, 10)

def test_probabilities_sum_to_one():
    p = model.predict(x_test[:5], verbose=0)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4)

def test_accuracy_above_98_percent():
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    assert acc > 0.98