import logging

logging.basicConfig(level=logging.INFO)
import argparse
import os
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import tensorflow as tf

from baselines.SingleHDR.dequantization_net import DequantizationNet
from baselines.SingleHDR.linearization_net import LinearizationNet
from baselines.SingleHDR.util import apply_rf
import numpy as np
import cv2
import glob

epsilon = 0.001

_clip = lambda x: tf.clip_by_value(x, 0, 1)

def build_graph(ldr, training):
    """Build the graph for the single HDR model.
    Args:
        ldr: [b, h, w, c], float32
        training: bool
    Returns:
        B_pred: [b, h, w, c], float32
    """

    # dequantization
    print('dequantize ...')
    dequantization_model = DequantizationNet(is_train=training)
    C_pred = _clip(dequantization_model(ldr))

    # linearization
    print('linearize ...')
    lin_net = LinearizationNet()
    # pred_invcrf = lin_net(C_pred, training)
    pred_invcrf = lin_net(C_pred)
    B_pred = apply_rf(C_pred, pred_invcrf)

    return B_pred

def build_session(root):
    """Build TF session and load models.
    Args:
        root: root path
    Returns:
        sess: TF session (dummy for compatibility)
    """
    return None


def dequantize_and_linearize(ldr_img, sess, graph, ldr, is_training):
    """Dequantize and linearize LDR image.
    Args:
        ldr_img: [H, W, 3], uint8
        sess: TF session (unused in TF2)
        graph: TF graph function
    Returns:
        linear_img: [H, W, 3], float32
    """
    ldr_val = np.flip(ldr_img, -1).astype(np.float32) / 255.0

    ORIGINAL_H = ldr_val.shape[0]
    ORIGINAL_W = ldr_val.shape[1]

    """resize to 64x"""
    if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
        RESIZED_H = int(np.ceil(float(ORIGINAL_H) / 64.0)) * 64
        RESIZED_W = int(np.ceil(float(ORIGINAL_W) / 64.0)) * 64
        ldr_val = cv2.resize(ldr_val, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC)

    padding = 32
    ldr_val = np.pad(ldr_val, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')

    print('inference ...')

    """run inference"""
    ldr_tensor = tf.constant([ldr_val], dtype=tf.float32)
    # is_training_tensor = tf.constant(False, dtype=tf.bool)
    lin_img = graph(ldr_tensor, training=False)
    
    lin_img = lin_img.numpy()
    """output transforms"""
    lin_img = np.flip(lin_img[0], -1)
    lin_img = lin_img[padding:-padding, padding:-padding]
    if ORIGINAL_H % 64 != 0 or ORIGINAL_W % 64 != 0:
        lin_img = cv2.resize(lin_img, dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC)

    return lin_img


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_imgs', type=str, default="/home/dev/IntrinsicHDR/inference_images/GenAI")
    parser.add_argument('--output_path', type=str, default="/home/dev/IntrinsicHDR/deq_lin_outputs")
    parser.add_argument('--root', type=str, default=".")
    parser.add_argument('--start_id',type=int, default=0)
    parser.add_argument('--end_id',type=int, default=None)
    parser.add_argument('--subfolder_structure',type=bool, default=True)
    
    args = parser.parse_args()

    # @tf.function
    def lin_graph(ldr, training=False):
        return build_graph(ldr, training=training)

    sess = build_session(args.root)

    # dummy_input = tf.constant(np.zeros((1, 64, 64, 3), dtype=np.float32))
    dummy_input = tf.zeros(shape=(1, 64, 64, 3), dtype=tf.float32)
    dummy_training = False
    _ = lin_graph(dummy_input, dummy_training)

    print('Loading checkpoints...')
    # checkpoint_path = args.root + '/baselines/SingleHDR/checkpoints/model.ckpt'
    checkpoint_path = args.root + '/baselines/SingleHDR/Liu_ckpt/model.ckpt'
    
    dequant_vars = [v for v in tf.compat.v1.trainable_variables() if 'Dequantization_Net' in v.name]
    if dequant_vars:
        dequant_checkpoint = tf.train.Checkpoint(**{v.name: v for v in dequant_vars})
        dequant_checkpoint.restore(checkpoint_path).expect_partial()
    
    lin_vars = [v for v in tf.compat.v1.trainable_variables() if 'crf_feature_net' in v.name or 'ae_invcrf_' in v.name]
    if lin_vars:
        lin_checkpoint = tf.train.Checkpoint(**{v.name: v for v in lin_vars})
        lin_checkpoint.restore(checkpoint_path).expect_partial()

    # get images
    if args.subfolder_structure:
        ldr_imgs = glob.glob(osp.join(args.test_imgs, '**', '*.png'))
        ldr_imgs.extend(glob.glob(osp.join(args.test_imgs, '**', '*.jpg')))
    else:
        ldr_imgs = glob.glob(osp.join(args.test_imgs, '*.png'))
        ldr_imgs.extend(glob.glob(osp.join(args.test_imgs, '*.jpg')))
        
    ldr_imgs = sorted(ldr_imgs)[args.start_id:args.end_id]


    ldr = None
    is_training = None

    for d, ldr_img_path in tqdm(enumerate(ldr_imgs), initial=args.start_id):
        ldr_img_name = osp.split(ldr_img_path)[-1][:-4]
        print(f"Processing image: {ldr_img_name}")
        # if ldr_img_name != "06327": continue
        
        ldr_dir = osp.dirname(ldr_img_path)
        target_dir = "/".join(ldr_dir.rsplit('/', 2)[1:])
        os.makedirs(osp.join(args.output_path, target_dir), exist_ok=True)
        
        # load img and preprocess
        ldr_img = cv2.imread(ldr_img_path)
        ldr_img = cv2.cvtColor(ldr_img,cv2.COLOR_BGR2RGB)

        # dequantize and linearize
        linear_img = dequantize_and_linearize(ldr_img, sess, lin_graph, ldr, is_training)

        # save linear image
        cv2.imwrite(osp.join(args.output_path, target_dir, ldr_img_name+'.exr'), 
                    cv2.cvtColor(linear_img,cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_EXR_COMPRESSION,1])

    print('Finished!')