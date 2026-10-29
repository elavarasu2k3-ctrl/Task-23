
import os, sys, time, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
n = int(sys.argv[1])
tf.config.threading.set_inter_op_parallelism_threads(n)
tf.config.threading.set_intra_op_parallelism_threads(n)
a = tf.random.normal((1024, 1024))
@tf.function
def f(a): return tf.reduce_sum(tf.matmul(a, a))
f(a).numpy()                                      # warm-up (tracing)
times = []
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(20): f(a).numpy()
    times.append(time.perf_counter() - t0)
print(json.dumps({"threads": n, "mean_s": sum(times)/len(times),
                  "matmuls_per_s": 20 / (sum(times)/len(times))}))
