import cv2
from pathlib import Path
import seaborn as sb
import tensorflow as tf
from matplotlib import pyplot as plt

from Settings import config
from Tensorflow.Architecture.ModelFeatureExtraction.inception_resnet_v1 import InceptionResNetV1
from utlis.argsparse import parser_test
from utlis.evalute import *


def preprocess(np_array):
    img = cv2.resize(np_array, (112, 112))
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def pipline(path_array):
    np_images = []
    for abspath_filename in path_array:
        np_image = cv2.imread(abspath_filename)
        np_preprocessed = preprocess(np_image)
        np_images.append(np_preprocessed)
    return np.asarray(np_images)


if __name__ == '__main__':
    # PARAMETER
    args = parser_test()

    # DATA
    pairs = read_pairs(pairs_filename=args.file_pair)

    paths, actual_issame = get_paths(args.folder_data, pairs)

    list_array = pipline(paths)

    # MODEL
    model = InceptionResNetV1(embedding_size=config.EMBEDDING_SIZE, name="InceptionResNetV1")

    # Loading checkpoint (if you have)
    path_checkpoint = Path('/Volumes/Ventoy/Data/Checkpoint/NEW_Checkpoint')
    checkpoint_path = tf.train.latest_checkpoint(path_checkpoint)
    print('[*] load ckpt from {}.'.format(checkpoint_path))
    model.load_weights(checkpoint_path)

    # EVALUATE
    distances, labels = evalute(embedding_size=config.EMBEDDING_SIZE,
                                batch_size=config.BATCH_SIZE,
                                model=model,
                                carray=list_array, issame=actual_issame)

    metrics = evaluate_lfw(distances=distances, labels=labels)

    txt = "Accuracy on Fujinet: {:.4f}+-{:.4f}\nPrecision {:.4f}+-{:.4f}\nRecall {:.4f}+-{:.4f}" \
          "\nROC Area Under Curve: {:.4f}\nBest distance threshold: {:.2f}+-{:.2f}" \
          "\nTAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
        np.mean(metrics['accuracy']),
        np.std(metrics['accuracy']),
        np.mean(metrics['precision']),
        np.std(metrics['precision']),
        np.mean(metrics['recall']),
        np.std(metrics['recall']),
        metrics['roc_auc'],
        np.mean(metrics['best_distances']),
        np.std(metrics['best_distances']),
        np.mean(metrics['tar']),
        np.std(metrics['tar']),
        np.mean(metrics['far']))

    title = 'Fujinet metrics'
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=15)
    fig.set_size_inches(14, 6)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    axes[0].set_title('distance histogram')
    sb.distplot(distances[labels == True], ax=axes[0], label='distance-true')
    sb.distplot(distances[labels == False], ax=axes[0], label='distance-false')
    axes[0].legend()

    axes[1].text(0.05, 0.3, txt, fontsize=20)
    axes[1].set_axis_off()
    plt.show()
