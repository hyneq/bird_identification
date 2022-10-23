#!/usr/bin/env python3

# This code heavily relies on the guide at https://medium.com/@chengweizhang2012/how-to-convert-trained-keras-model-to-a-single-tensorflow-pb-file-and-make-prediction-4f7337fc96af

import os
import argparse

from keras import backend as K
import tensorflow as tf

# from https://gist.github.com/Tony607/40d9fe0c83603147ce71905da2d5a04a#file-freeze_session-py
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def keras_to_tensorflow(keras_name, tf_name=None):

    if tf_name is None:
        tf_name = os.path.splitext(keras_name[0]) + ".pb"

    # from https://gist.github.com/Tony607/95c01021ad2cc76f899be64fd282577c#file-tensor_names-py
    from tensorflow.keras.models import load_model
    model = load_model(keras_name)
    print(model.outputs)
    print(model.inputs)

    # from https://gist.github.com/Tony607/40d9fe0c83603147ce71905da2d5a04a#file-freeze_session-py
    frozen_graph = freeze_session(K.get_session(),
                                output_names=[out.op.name for out in model.outputs])

    # from https://gist.github.com/Tony607/494fc5a5c102b3626de6adf22ed8e02d#file-write_graph-py
    tf.train.write_graph(frozen_graph, tf_dirname, tf_filename, as_text=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("keras_name", help="Name of Keras model file")
    parser.add_argument("tf_name", default=None, help="Name of Tensorflow model file")
    args = parser.parse_args()

    keras_to_tensorflow(args.keras_name, args.tf_name)