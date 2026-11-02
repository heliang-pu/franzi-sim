"""The policy server's wire format, shared by fineart.serve and isaac.policy.

Messages go over multiprocessing.connection on localhost as plain Python values
only; an array travels as {"dtype", "shape", "data": bytes}. The Isaac side has
numpy 1.26 and the server numpy 2, which can't unpickle each other's arrays.
"""

import numpy as np

HOST = "127.0.0.1"
PORT = 18780
AUTHKEY = b"inovance-fineart-policy"  # keeps stray local connections out
# Control ticks between closed-loop subtask queries (3 s), and the lag of the
# memory recipe's "Previous subtask" (fineart.recipe): in training the label
# that many ticks back, in closed loop the subtask of the previous query.
SUBTASK_EVERY = 75
NO_MEMORY = "none"  # the memory before the first query, and in the first 3 s


def pack(array):
    array = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": array.tobytes(),
    }


def unpack(message):
    return np.frombuffer(message["data"], dtype=np.dtype(message["dtype"])).reshape(
        message["shape"]
    )


def address(text):
    """host:port (or a port alone) as a multiprocessing.connection address."""
    host, _, port = text.rpartition(":")
    return (host or HOST, int(port))
