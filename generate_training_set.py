import os
import argparse
import numpy as np
import cv2
from scipy import ndimage
from multiprocessing import Process

def get_sub_path(path):
    sub_path = []
    if isinstance(path, list):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p, file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path, file))
    return sub_path

def resize(input):
    dimension = input.shape
    result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
    return result

def resize_cv2(input):
    return cv2.resize(input, (256, 256), interpolation=cv2.INTER_AREA)

def std(input):
    if input.max() == 0:
        return input
    return (input - input.min()) / (input.max() - input.min())

def save_npy(out_list, save_path, name):
    output = np.array(out_list)
    output = np.transpose(output, (1, 2, 0))
    np.save(os.path.join(save_path, name), output)

def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]

def pack_data(args, name_list, read_feature_list, read_label_list, save_path):

    feature_save_path = os.path.join(args.save_path, args.task, 'feature')
    label_save_path = os.path.join(args.save_path, args.task, 'label')

    os.makedirs(feature_save_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)

    for name in name_list:
        base = os.path.basename(name)

        # FEATURES
        out_feature_list = []
        for feature_name in read_feature_list:
            feature = np.load(os.path.join(args.data_path, feature_name, base))

            if args.task == 'IR_drop':
                if feature_name.endswith('power_t'):
                    for i in range(20):
                        slice = feature[i,:,:]
                        out_feature_list.append(std(resize_cv2(slice)))
                else:
                    feature = std(resize_cv2(feature.squeeze()))
                    out_feature_list.append(feature)
            else:
                raise ValueError('Not implemented')

        save_npy(out_feature_list, feature_save_path, base)

        # LABELS
        out_label_list = []
        for label_name in read_label_list:
            label = np.load(os.path.join(args.data_path, label_name, base))
            label = np.squeeze(label)
            label = np.clip(label, 1e-6, 50)
            label = (np.log10(resize_cv2(label)) + 6) / (np.log10(50) + 6)
            out_label_list.append(label)

        save_npy(out_label_list, label_save_path, base)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None)
    parser.add_argument("--data_path", default=".")
    parser.add_argument("--save_path", default="./training_set")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.task != 'IR_drop':
        raise ValueError("For now only IR_drop supported")

    feature_list = ['power_i', 'power_s', 'power_sca', 'power_all', 'power_t']

    label_list = ['IR_drop']

    name_list = get_sub_path(os.path.join(args.data_path, feature_list[0]))

    print("processing %s files" % len(name_list))
    save_path = os.path.join(args.save_path, args.task)

    nlist = divide_list(name_list, 1000)
    process = []

    for lst in nlist:
        p = Process(target=pack_data, args=(args, lst, feature_list, label_list, save_path))
        process.append(p)

    for p in process:
        p.start()

    for p in process:
        p.join()
