import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import os, sys
import progressbar
import matplotlib.pyplot as plt
import argparse

def load_features(feature_dir):
    """Load in all features from a folder. Features are assumed to be .npy files"""
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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Classify image features extracted from CNN.')
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

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    source_dir = args.source_data_path
    target_dir = args.target_data_path
    x_cache = args.feature_cache_file
    y_cache = args.label_cache_file

    # First try to load cached features
    if (x_cache is not None) and (y_cache is not None) and \
            os.path.exists(x_cache) and os.path.exists(y_cache):
        x = np.load(x_cache)
        y = np.load(y_cache)
    else:
        source_features = load_features(source_dir)
        target_features = load_features(target_dir)
        x = np.vstack([source_features, target_features])

        n_source = source_features.shape[0]
        n_target = target_features.shape[0]
        source_label = np.ones(n_source)
        target_label = np.zeros(n_target)
        y = np.hstack([source_label, target_label])

        np.save(x_cache, y)
        np.save(y_cache, x)

    print "Loaded %i features!" % (x.shape[0])

    # Split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    # Create and fit the model
    svm = svm.LinearSVC()
    svm.fit(x_train, y_train)

    score = svm.score(x_test, y_test)
    print "Score: %0.2f" % score

    # Create the plot
    distances = svm.decision_function(x_test)

    source_idx = y_test == 1
    target_idx = y_test == 0

    n, bins, patches = plt.hist(
            [distances[source_idx], distances[target_idx]],
            bins=40,
            label=["Source Domain", "Target Domain"])

    plt.xlabel("Distance to decision hyperplane")
    plt.ylabel("Count")
    plt.title("Distributions of pool5 layer over target and source datasets")
    plt.xlim(-5, 5)
    plt.legend()
    plt.grid(True)

    plt.show()

