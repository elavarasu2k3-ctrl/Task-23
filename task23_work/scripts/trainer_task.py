
import os, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, keras
p = argparse.ArgumentParser()
p.add_argument("--n_layers", type=int, default=2)
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--job-dir", dest="job_dir", required=True)
p.add_argument("--epochs", type=int, default=3)
a = p.parse_args()
d = np.load(r"C:\Users\HP\Documents\Task23\task23_work\mnist_subset.npz")
tf.random.set_seed(0)
model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten()] +
                         [keras.layers.Dense(64, activation="relu") for _ in range(a.n_layers)] +
                         [keras.layers.Dense(10, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=a.momentum))
model.fit(d["X_train"][:10000], d["y_train"][:10000], epochs=a.epochs, batch_size=64, verbose=0,
          validation_data=(d["X_valid"], d["y_valid"]),
          callbacks=[keras.callbacks.TensorBoard(log_dir=a.job_dir)])   # <- all the service needs
