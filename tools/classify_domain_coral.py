import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import os, sys
import progressbar
import matplotlib.pyplot as plt
import argparse

def load_features(source_feature_dir, target_feature_dir, source_cache=None, target_cache=None):
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
        names = [os.path.join(feature_dir, n) for n in os.listdir(feature_dir)]
        file_paths = [n for n in names if os.path.isfile(n)]
        num_files = len(file_paths)

        bar = progressbar.ProgressBar(redirect_stdout=True)
        feature_list = [None] * num_files
        for i, file_path in bar(enumerate(file_paths), max_value=num_files):
            if not file_path.endswith('.npy'):
                print "Ignoring %s" % file_path
                continue

            feature = np.load(file_path)
            feature_list[i] = feature

        # Concat into one array
        features = np.vstack(feature_list)

        return features

    # First try to load cached features
    if (source_cache is not None) and (target_cache is not None) and \
            os.path.exists(source_cache) and os.path.exists(target_cache):
        x = np.load(source_cache)
        y = np.load(target_cache)

        print "Loaded cached features"
        return x, y

    source_features = _load_features_helper(source_feature_dir)
    if source_cache is not None:
        np.save(source_cache, source_features)

    target_features = _load_features_helper(target_feature_dir)
    if target_cache is not None:
        np.save(target_cache, target_features)

    return source_features, target_features

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
    parser.add_argument('coral_matrix_path',
                        help='Folder with precomputed CORAL alignment matrix.',
                        type=str)
    parser.add_argument('--source_cache_file', dest='source_cache_file',
                        help='File where source features will be cached',
                        default=None, type=str)
    parser.add_argument('--target_cache_file', dest='target_cache_file',
                        help='File where target features will be cached',
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
    source_cache = args.source_cache_file
    target_cache = args.target_cache_file
    coral_matrix_path = args.coral_matrix_path

    # Load features
    source_features, target_features = load_features(source_dir, target_dir, source_cache, target_cache)

    # Apply CORAL transform
    A = np.load(coral_matrix_path)
    target_features = np.dot(target_features, A.T)

    num_source = source_features.shape[0]
    num_target = target_features.shape[0]
    print "Loaded %i source features" % num_source
    print "Loaded %i target features" % num_target

    x = np.vstack((source_features, target_features))

    y_source = np.ones(num_source)
    y_target = np.zeros(num_target)
    y = np.hstack((y_source, y_target))


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

