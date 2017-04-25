import _init_paths
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import os, sys
import progressbar
import argparse

from networks.classify import ClassifyDomainTrain, ClassifyDomainTest

def snapshot_npy(sess, output_dir, iter):
    """Take a snapshot of the network to a .npy file."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = {}
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for i in range(len(all_variables)):
        variable_name = all_variables[i].name

        parts = variable_name.split('/');
        if (len(parts) == 1):
            continue
        if (len(parts) == 2):
            scope_name = parts[0]
        if (len(parts) == 3):
            scope_name = parts[0] + '/' + parts[1]
            vari_name_temp = parts[-1].split(':')[0]
            if (vari_name_temp != "weights" and vari_name_temp != "biases"):
                continue
        if (len(parts) > 3):
            continue
        vari_name = parts[-1].split(':')[0]
        if (scope_name not in data.keys()):
            data[scope_name] = {}
        data[scope_name][vari_name] = sess.run(all_variables[i])

    filename = output_dir + '/' + str(iter + 1) + '.npy'
    np.save(filename,data);

    print 'Wrote snapshot to: {:s}'.format(filename)


def load_features(source_dir, target_dir, feature_cache=None, label_cache=None):
    """Load in all features from the given folders. Features are assumed to be .npy files"""

    def _load_features_helper(feature_dir):
        """
        Helper function to handle the actual loading.
        Loads all .npy files in the given folder.
        """
        if not os.path.exists(feature_dir):
            raise IOError("Feature directory %s doesn't exist!" % feature_dir)

        print "Loading features from folder '%s'" % feature_dir

        # Iterate over all files in given directory
        features = None
        bar = progressbar.ProgressBar()
        for file in bar(os.listdir(feature_dir)):
            file_path = os.path.join(feature_dir, file)

            if not file.endswith('.npy'):
                print "Ignoring %s" % file_path
                continue

            feature = np.load(file_path)
            if feature.ndim != 2:
                feature = np.mean(feature, axis=(1,2))

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

        return features

    # First try to load cached features
    if (feature_cache is not None) and (label_cache is not None) and \
            os.path.exists(feature_cache) and os.path.exists(label_cache):
        x = np.load(feature_cache)
        y = np.load(label_cache)

        print "Loaded cached features"
        return x, y

    source_features = load_features(source_dir)
    target_features = load_features(target_dir)
    x = np.vstack([source_features, target_features])

    n_source = source_features.shape[0]
    n_target = target_features.shape[0]
    source_label = np.ones(n_source)
    target_label = np.zeros(n_target)
    y = np.hstack([source_label, target_label])

    if feature_cache is not None:
        np.save(feature_cache, x)
    if label_cache is not None:
        np.save(label_cache, y)

    return x, y

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
            description=
            """
            Classify image features extracted from CNN.
            Uses a shallow MLP.
            """)
    parser.add_argument('source_data_path',
                        help='Folder with source domain features',
                        type=str)
    parser.add_argument('target_data_path',
                        help='Folder with target domain features',
                        type=str)
    parser.add_argument('--feature_cache_file', dest='feature_cache_file',
                        help='File where loaded features will be cached',
                        default=None, type=str)
    parser.add_argument('--label_cache_file', dest='label_cache_file',
                        help='File where computed labels will be cached',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train_net(sess, net, x, y, net_test, x_test, y_test, num_epochs=10):
    # Set up optimizer
    global_step = tf.Variable(0, trainable=False)
    output = net.get_output('conf_score')
    label_tf = net.get_output('label')
    loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label_tf))
    optim = tf.train.AdamOptimizer(0.001).minimize(loss_op, global_step=global_step)

    # Set up summary writer
    tf.summary.scalar("loss", loss_op)
    summary_merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join("tensorboard_test", "train"), sess.graph)

    # Set up model saver
    saver = tf.train.Saver(max_to_keep=100)

    # Initialize
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Train for 10 epochs
    batch_size = 124
    num_batches = num_samples / batch_size
    for epoch in xrange(num_epochs):
        for iter in xrange(num_batches):
            samples = range(batch_size*iter, batch_size*(iter+1))
            batch_x = x[samples, ...]
            batch_y = y[samples, ...]

            feed_dict = {net.data: batch_x, net.label: batch_y}

            loss_val, _ = sess.run([loss_op, optim], feed_dict=feed_dict)

            if (iter+1) % 25 == 0:
                summary_results = sess.run(summary_merged, feed_dict=feed_dict)
                train_writer.add_summary(summary_results, global_step.eval())

        print "epoch %i loss: %0.3f" % (epoch+1, loss_val)
        test_net(sess, net_test, x_test, y_test)
        filename = os.path.join("checkpoints", "model")
        saver.save(sess, filename, epoch+1)

        snapshot_npy(sess, "checkpoints", epoch)

    print "Done training!"



def test_net(sess, net, x, y):
    # Set up accuracy measurement op
    output = net.get_output('conf_score')
    pred = net.get_output('conf_prob')
    label_tf = net.get_output('label')

    correct = tf.nn.in_top_k(pred, label_tf, 1)
    acc_op = tf.reduce_mean(tf.to_float(correct))
    loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label_tf))

    # Treat all of data as one batch and hope the gpu can handle it
    feed_dict = {net.data: x, net.label: y}
    acc, loss = sess.run([acc_op, loss_op], feed_dict=feed_dict)

    print "Validation loss: %0.3f, accuracy: %0.3f" % (loss, acc)

    #for iter in xrange(num_samples):
    #    batch_x = x[iter, ...]
    #    batch_y = y[iter, ...]
    #    feed_dict = {net.data: batch_x, net.label: batch_y}

    #    acc  = sess.run([acc_op], feed_dict=feed_dict)[0]


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    source_dir = args.source_data_path
    target_dir = args.target_data_path
    x_cache = args.feature_cache_file
    y_cache = args.label_cache_file

    x, y = load_features(source_dir, target_dir, x_cache, y_cache)

    print "Loaded %i features!" % (x.shape[0])

    # Split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    num_samples = x_train.shape[0]

    with tf.Session() as sess:
        # Set up network
        net = ClassifyDomainTrain()
        net_test = ClassifyDomainTest()

        train_net(sess, net, x_train, y_train, net_test, x_test, y_test, num_epochs=100)

        #test_net(sess, net, x_test, y_test)
