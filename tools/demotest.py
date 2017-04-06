import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import pdb as pdb
namepath = './kittitrain_test.txt'
filepath = '../data/training/image_2/';
imgpath = './results/img1_out/'
txtpath = './results/txt1_out/'
CLASSES = ('__background__','Car','Pedestrian')
           # 'aeroplane', 'bicycle', 'bird', 'boat',
           # 'bottle', 'bus', 'car', 'cat', 'chair',
           # 'cow', 'diningtable', 'dog', 'horse',
           # 'motorbike', 'person', 'pottedplant',
           # 'sheep', 'sofa', 'train', 'tvmonitor')


# Alternative list of classes. Each of these we could expect to see in our data.
#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)
    #pdb.set_trace()
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

def demo_no_plot(sess, net, image_name):
    """
    Detect object classes in an image using pre-computed object proposals.
    Returns results instead of displaying.
    """

    # Load the demo image
    im_file = os.path.join(filepath, image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    res = np.array([])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        cls_group = np.ones(len(keep)) * cls_ind;
        dets = dets[keep, :]
        dets = np.hstack((dets, cls_group[:, np.newaxis])).astype(np.float32)
        inds = np.where(dets[:, -2] >= CONF_THRESH)[0]
        dets = dets[inds, :]
        if (cls_ind == 1):
            res = dets
        else:
            res = np.vstack((res, dets))
    return res;


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--text_only', dest='text_only',
                        help="If set, don't save images",
                        default=False, type=bool)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # init session

    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)

    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Create directories
    if (not os.path.exists(imgpath)):
        os.mkdir(imgpath)
    if (not os.path.exists(txtpath)):
        os.mkdir(txtpath)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    testFiles = []
    with open(namepath, 'r') as f:
        for line in f:
            name = line.strip()
            testFiles.append(name)

    numFiles = len(testFiles)
    for i in range(numFiles):
        im_name = testFiles[i] + '.png'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        res = demo_no_plot(sess, net, im_name)
        numBox = res.shape[0]
        tofile_txt = txtpath + testFiles[i] + '.txt'
        tofile_img = imgpath + testFiles[i] + '.jpg'
        with open(tofile_txt, 'w') as ftxt:
            for i in range(numBox):
                ftxt.write(CLASSES[int(res[i][5])] + ' ' + str(res[i][0]) + ' ' + str(res[i][1]) + ' ' +
                        str(res[i][2]) + ' ' + str(res[i][3]) + ' '  + str(res[i][4]) + '\n')

        if not cfg.text_only:
            im = cv2.imread(filepath + im_name)

            for i in range(numBox):
                cv2.rectangle(im,(int(res[i][0]),int(res[i][1])),(int(res[i][2]),int(res[i][3])),(0,255,0),2)
                cv2.rectangle(im,(int(res[i][0]),int(res[i][1]-20)),(int(res[i][2]),int(res[i][1])),(125,125,125),-1)
                cv2.putText(im,CLASSES[int(res[i][5])] + ' : %.2f' % res[i][4],(int(res[i][0])+5,int(res[i][1])-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            cv2.imwrite(tofile_img,im)
