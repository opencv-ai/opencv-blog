import sys
sys.path.insert(0, 'src')
import transform
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.python.framework import graph_io


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    parser.add_argument('--out_graph_name', type=str,
                        default="inference_graph.pb")

    return parser.parse_args()


def main(checkpoint_path, input_shape, out_graph_name):
    # Init graph and session to be used
    g = tf.Graph()
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    with g.as_default(), g.device('/cpu'), tf.compat.v1.Session(config=soft_config) as sess:
        # Placeholder variable for graph input
        img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=input_shape,
                                                   name='img_placeholder')
        # The model from the repo
        transform.net(img_placeholder)

        # Restore model from checkpoint
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint_path)

        # Freeze graph from the session.
        # "add_37" is the actual last operation of graph
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["add_37"])
        # Write frozen graph to a file
        graph_io.write_graph(frozen, './', out_graph_name, as_text=False)
        print(f'Frozen graph {out_graph_name} is saved!')


if __name__ == "__main__":
    args = parse_args()
    input_shape = (1, 720, 1024, 3)
    main(args.checkpoint, input_shape, args.out_graph_name)
