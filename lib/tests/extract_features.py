import numpy as np
import cv2
import os
from utils.timer import Timer
import random
import progressbar

FEATURE_LAYER = "fc7"

def extract_feature(sess, net, im):
    """Extract image feature from the feature layer layer for single image."""
    # Reshape to be a 4-d tensor of shape (batch, height, width, channels)
    if im.ndim != 4:
        im = im.reshape(1, *im.shape)

    im_info = np.array(
        [[im.shape[1], im.shape[2], 1]],
        dtype=np.float32)
    feature_op = net.get_output(FEATURE_LAYER)
    feed_dict = {net.data: im, net.im_info: im_info}

    feature = sess.run([feature_op], feed_dict=feed_dict)[0]
    #feature = np.mean(feature, (1,2))

    return feature

def extract_features(sess, net, imdb, output_dir, max_images=None):
    """Extract image feature from the pool5 layer for entire image database."""
    num_images = len(imdb.image_index)
    image_indices = imdb._load_image_set_index()

    # timers
    _t = {'feature' : Timer(), 'misc' : Timer(), 'imread' : Timer()}

    image_list = xrange(num_images)
    if max_images:
        random.seed(42)
        image_list = random.sample(list(image_list), max_images)
        num_images = max_images

    bar = progressbar.ProgressBar(redirect_stdout=True)
    for i, index in bar(enumerate(image_list), max_value=num_images):
        # Load an image
        _t['imread'].tic()
        im = cv2.imread(imdb.image_path_at(index))
        _t['imread'].toc()

        # Run extraction
        _t['feature'].tic()
        feature = extract_feature(sess, net, im)
        _t['feature'].toc()

        # Save feature to disc
        _t['misc'].tic()
        out_path = os.path.join(output_dir, image_indices[i])
        np.save(out_path, feature)
        _t['misc'].toc()

        print 'feature: {:d}/{:d} {:.3f}s {:.3f}s, {:.3f}s' \
              .format(i + 1, num_images, _t['feature'].average_time,
                      _t['misc'].average_time, _t['imread'].average_time)

