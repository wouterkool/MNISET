import os
import numpy as np
import itertools

DATASET_URL = 'https://github.com/wouterkool/MNISET/raw/master/data/MNISET.npz'


def load_mniset(datadir='data', dataset_url=DATASET_URL):
    os.makedirs(datadir, exist_ok=True)
    dataset_filename = os.path.join(datadir, os.path.split(dataset_url)[-1])
    if not os.path.isfile(dataset_filename):
        # download
        from urllib import request
        request.urlretrieve(dataset_url, dataset_filename)
    return np.load(dataset_filename)


def split_classes(y):
    # See plot_classes
    row, col = y // 9, y % 9
    qty, fill = row // 3, row % 3
    color, shape = col // 3, col % 3
    return qty, fill, color, shape


def make_labels(*args):
    return np.array(["-".join(feats) for feats in list(itertools.product(*args))])


def get_split(mniset, split='train'):
    if split == 'train':
        x, y = mniset['x_train'], mniset['y_train']
    elif split == 'test':
        x, y = mniset['x_test'], mniset['y_test']
    else:
        assert False, f"Unkown split {split}"
    return x, y


def create_classes(mniset, keep_feature_mask, defaults=[1, 2, 0, 2]):
    
    defaults = np.array(defaults) # 2 solid red squiggle
    
    num_orig_classes = 3 ** len(keep_feature_mask)
    # We can split the full classes into individual classes for features
    feat_classes = np.column_stack(split_classes(np.arange(num_orig_classes)))
    
    # Keep each original class if we want to keep it or it is the default for a feature
    keep_class_mask = ((feat_classes == defaults[None, :]) | keep_feature_mask[None, :]).all(-1)
    
    # Create feature labels for classes to keep and filter label imgs
    feature_names = mniset['feature_names'][keep_feature_mask]
    feature_labels = mniset['feature_labels'][keep_feature_mask]
    labels = make_labels(*feature_labels)
    label_imgs = mniset['label_imgs'][keep_class_mask]
    return labels, label_imgs, feature_names, feature_labels


def combine_classes(y_feat):
    # Convert 'base 3' representation to single decimal
    return ((3 ** np.arange(y_feat.shape[-1])[::-1])[None, :] * y_feat).sum(-1)


def extract_dataset(mniset, split='train', qty=True, fill=True, color=True, shape=True):
    
    keep_feature_mask = np.array([qty, fill, color, shape])
    assert keep_feature_mask.any(), "Need to have at least one feature in class"
    
    x, y = get_split(mniset, split)
    
    labels, label_imgs, feature_names, feature_labels = create_classes(mniset, keep_feature_mask)
    
    # Get individual features based on labels and keep only those which we want to keep
    y_feat = np.column_stack(split_classes(y))[:, keep_feature_mask]
    # Combine the classes for the features
    y = combine_classes(y_feat)
    
    # Convert to float 0-1
    return x / 255., y, labels, label_imgs / 255.


def extract_grayscale_dataset(mniset, split='train', qty=True, fill=True, color=False, shape=True):
    # Converts images to grayscale and has color feature disabled by default
    x, y, labels, label_imgs = extract_dataset(
        mniset, split, qty, fill, color, shape
    )
    # Convert to grayscale
    x, label_imgs = x.mean(-1), label_imgs.mean(-1)
    return x, y, labels, label_imgs
