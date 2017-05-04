import numpy as np

import os, sys
import progressbar
import argparse

from time import time

def load_features(source_dir, target_dir, source_cache=None, target_cache=None):
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

    source_features = _load_features_helper(source_dir)
    if source_cache is not None:
        np.save(source_cache, source_features)

    target_features = _load_features_helper(target_dir)
    if target_cache is not None:
        np.save(target_cache, target_features)

    return source_features, target_features

def compute_transform_matrix(source_features, target_features, num_features_to_use=None):
    # Test on smaller subset
    if num_features_to_use is not None:
        print "Using only %i features for covariance computation" % num_features_to_use
        source_features = source_features[:num_features_to_use,:]
        target_features = target_features[:num_features_to_use,:]

    tic = time()
    print "Computing source covariance..."
    cov_source = np.cov(source_features, rowvar=False)
    print "Computing target covariance..."
    cov_target = np.cov(target_features, rowvar=False)
    toc = time()
    print "Covariance computation took %0.3f seconds" % (toc - tic)

    tic = time()
    print "Computing SVD of source covariance"
    U_source, S_source, _ = np.linalg.svd(cov_source)
    print "Computing SVD of target covariance"
    U_target, S_target, _ = np.linalg.svd(cov_target)
    toc = time()
    print "SVD computation took %0.3f seconds" % (toc - tic)

    tic = time()
    # Compute rank
    s_thresh = 1e-9
    r_source = np.sum(S_source > s_thresh)
    r_target = np.sum(S_target > s_thresh)
    r = min(r_source, r_target)

    # Reduce rank as necessary
    U_target = U_target[:,:r]
    S_target = S_target[:r]

    # Turn vectors back into diag matrices, apply sqrt and pinv as necessary
    S_source_pinv_sqrt = np.diag(S_source**(-0.5))
    S_target_sqrt = np.diag(S_target**(0.5))

    # Finally, compute A, the optimal transform matrix
    cov_source_hat = np.dot(U_source, np.dot(S_source_pinv_sqrt, U_source.T))
    cov_target_hat = np.dot(U_target, np.dot(S_target_sqrt, U_target.T))

    A_star = np.dot(cov_source_hat, cov_target_hat)
    toc = time()
    print "The rest of the computation took %0.3f seconds" % (toc - tic)

    # Test improvement
    #cov_source_new = np.dot(A_star.T, np.dot(cov_source, A_star))
    #norm_before = np.linalg.norm(cov_source - cov_target, 'fro')
    #norm_after = np.linalg.norm(cov_source_new - cov_target, 'fro')

    #print "Norm reduced from %0.5f to %0.5f!" % (norm_before, norm_after)

    return A_star


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

    source_features, target_features = load_features(source_dir, target_dir, source_cache, target_cache)

    print "Loaded %i source features" % (source_features.shape[0])
    print "Loaded %i target features" % (target_features.shape[0])

    A = compute_transform_matrix(source_features, target_features)
    np.save(os.path.join('data','pretrain_model', 'coral.npy'), A)
