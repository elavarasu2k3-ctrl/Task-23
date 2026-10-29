
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"; os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU workers
import numpy as np, tensorflow as tf, keras
idx = json.loads(os.environ["TF_CONFIG"])["task"]["index"]
def log(msg): print(f"[worker {idx} {time.strftime('%H:%M:%S')}] {msg}", flush=True)
log("process started")
comm = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.RING)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=comm)
log(f"strategy created, replicas in sync = {strategy.num_replicas_in_sync}")
d = np.load(r"C:\Users\HP\Documents\Task23\task23_work\mnist_subset.npz")
X, y = d["X_train"][:8000], d["y_train"][:8000]
PER_REPLICA = 64
GLOBAL = PER_REPLICA * strategy.num_replicas_in_sync
opts = tf.data.Options()               # each worker reads every 2nd element of the dataset
opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
ds = (tf.data.Dataset.from_tensor_slices((X, y)).shuffle(8000, seed=0)
      .batch(GLOBAL, drop_remainder=True).with_options(opts))
dist_ds = strategy.experimental_distribute_dataset(ds)
with strategy.scope():
    tf.random.set_seed(0)
    model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                              keras.layers.Dense(128, activation="relu"), keras.layers.Dense(10, activation="softmax")])
    optimizer = keras.optimizers.SGD(learning_rate=0.05)
    optimizer.build(model.trainable_variables)
    loss_obj = keras.losses.SparseCategoricalCrossentropy(reduction="none")
@tf.function
def train_step(batch):
    def step_fn(inputs):
        x, yb = inputs
        with tf.GradientTape() as tape:
            loss = tf.nn.compute_average_loss(loss_obj(yb, model(x, training=True)), global_batch_size=GLOBAL)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))   # gradients all-reduced across workers
        return loss
    return strategy.reduce(tf.distribute.ReduceOp.SUM, strategy.run(step_fn, args=(batch,)), axis=None)
log("entering training loop - the first step blocks until every worker is up (AllReduce)")
for epoch in range(3):
    total, n = 0.0, 0
    for batch in dist_ds:
        total += float(train_step(batch)); n += 1
    log(f"epoch {epoch+1} done, mean loss={total/n:.4f}, steps={n}")
log("finished")
