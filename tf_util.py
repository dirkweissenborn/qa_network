import tensorflow as tf


def batch_dot(t1, t2):
    t1_e = tf.expand_dims(t1, 1)
    t2_e = tf.expand_dims(t2, 2)
    return tf.squeeze(tf.batch_matmul(t1_e, t2_e), [1, 2])


# compute all participating tensors in forward pass
def get_tensors(output_tensors, input_tensors, include_out=True, current_path=None):
    res = set()
    for o in output_tensors:
        if o not in input_tensors:  # we do not want to add inputs
            current_new = set()
            if include_out:
                current_new.add(o)  # we do not add o directly to res
            if current_path:
                current_new = current_new.union(current_path)
            res = res.union(get_tensors(o.op.inputs, input_tensors, True, current_new))
        else:
            # only keep paths leading to inputs
            res = res.union(current_path)
    return res