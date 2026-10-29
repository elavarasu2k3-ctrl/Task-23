
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
mode = sys.argv[1]
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print(json.dumps({"mode": mode, "result": "no GPU visible to this process"})); sys.exit(0)
if mode == "growth":
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
elif mode == "limit":
    for g in gpus: tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
elif mode == "split":
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)] * 2)
a = tf.random.normal((2048, 2048)); b = tf.matmul(a, a).numpy()
info = tf.config.experimental.get_memory_info("GPU:0")
print(json.dumps({"mode": mode, "visible_physical": [g.name for g in gpus],
                  "logical": [d.name for d in tf.config.list_logical_devices("GPU")],
                  "current_MB": info["current"] / 2**20, "peak_MB": info["peak"] / 2**20}))
