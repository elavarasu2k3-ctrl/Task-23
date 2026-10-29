
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
m = tf.saved_model.load(r"C:\Users\HP\Documents\Task23\task23_work\serving\pyfunc_model\0001")
f = m.signatures["serving_default"]
try:
    out = f(tf.constant(np.zeros((1, 28, 28), np.float32)))
    print("UNEXPECTED: call worked", out)
except Exception as e:
    print("Step 2 - call in a fresh process FAILED:", type(e).__name__)
    print(str(e).splitlines()[0][:300])
