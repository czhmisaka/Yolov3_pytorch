import os
import numpy as np
import xml.dom.minidom


def read_train_val():
    if os.path.exists('train.txt'):
        os.remove('train.txt')
    res = []
    txt_paths = ['VOCdevkit/VOC2012/ImageSets/Main/train.txt',
                 'VOCdevkit/VOC2012/ImageSets/Main/val.txt',
                 'VOCdevkit/VOC2007/ImageSets/Main/train.txt',
                 'VOCdevkit/VOC2007/ImageSets/Main/val.txt']

    img_path = ['VOCdevkit/VOC2012/JPEGImages',
                'VOCdevkit/VOC2012/JPEGImages',
                'VOCdevkit/VOC2007/JPEGImages',
                'VOCdevkit/VOC2007/JPEGImages']

    ann_path = ['VOCdevkit/VOC2012/Annotations',
                'VOCdevkit/VOC2012/Annotations',
                'VOCdevkit/VOC2007/Annotations',
                'VOCdevkit/VOC2007/Annotations']

    for txt, img, ann in zip(txt_paths, img_path, ann_path):
        with open(txt, 'r') as f:
            for line in f:
                res.append(line.strip('\n'))

        with open('train.txt', 'a') as f:
            for name in res:
                a_path = ''.join([ann, '/', name, '.xml'])
                i_path = ''.join([img, '/', name, '.jpg'])
                if os.path.exists(a_path) and os.path.exists(i_path):
                    f.write(''.join([a_path, ' ', i_path, '\r\n']))


def read_test():
    if os.path.exists('test.txt'):
        os.remove('test.txt')
    res = []

    txt_path = 'VOCdevkit/VOC2007_TEST/ImageSets/Main/test.txt'
    img_path = 'VOCdevkit/VOC2007_TEST/JPEGImages'
    ann_path = 'VOCdevkit/VOC2007_TEST/Annotations'

    with open(txt_path, 'r') as f:
        for line in f:
            res.append(line.strip('\n'))

    with open('test.txt', 'a') as f:
        for name in res:
            a_path = ''.join([ann_path, '/', name, '.xml'])
            i_path = ''.join([img_path, '/', name, '.jpg'])
            if os.path.exists(a_path) and os.path.exists(i_path):
                f.write(''.join([a_path, ' ', i_path, '\r\n']))


def get_all_wh(txt, img_size=416):
    """
    从 txt 文件中读取 xml 位置，并读取所有 xml
    取出所有图片中真实框的长宽信息
    并将他们以图片缩放的对应倍数缩放至正确大小
    :param txt: 图片记录文件的path
    :param img_size: 图片要缩放的尺寸
    :return: 所有图片中所有真实框的长宽
    """
    ann_list = []
    with open(txt, 'r') as f:
        for line in f:
            ann_list.append(line.split(' ')[0])

    all_size = []
    for ann in ann_list:
        DOMTree = xml.dom.minidom.parse(ann)
        annotation = DOMTree.documentElement
        size = annotation.getElementsByTagName('size')[0]
        w = int(size.getElementsByTagName('width')[0].childNodes[0].data)
        h = int(size.getElementsByTagName('height')[0].childNodes[0].data)
        ratio = max(w, h) / img_size

        objs = annotation.getElementsByTagName("object")
        for obj in objs:
            bbox = obj.getElementsByTagName('bndbox')[0]
            xmin = int(bbox.getElementsByTagName('xmin')[0].childNodes[0].data)
            xmax = int(bbox.getElementsByTagName('xmax')[0].childNodes[0].data)
            ymin = int(bbox.getElementsByTagName('ymin')[0].childNodes[0].data)
            ymax = int(bbox.getElementsByTagName('ymax')[0].childNodes[0].data)

            obj_w = (xmax - xmin) / ratio
            obj_h = (ymax - ymin) / ratio
            all_size.append((obj_w, obj_h))
    return all_size


def iou(sizes, cluster):
    """
    计算所有真实框与聚类中心的 IoU
    :param sizes: 真实框的长宽信息
    :param cluster: 聚类中心的长宽信息
    :return: IoU
    """
    intersection = np.minimum(sizes[:, 0], cluster[0]) * np.minimum(sizes[:, 1], cluster[1])
    union = sizes[:, 0] * sizes[:, 1] + cluster[0] * cluster[1] - intersection + 1e-10
    return intersection / union


def anchor_box_k_mean(img_size=416, k=9):
    """
    使用 k-means 将真实框进行聚类
    :param img_size: 训练时图片大小
    :param k: 聚类中心个数
    :return: 聚类中心
    """
    all_size = get_all_wh('train.txt', img_size)

    res = np.array(all_size)
    rows = res.shape[0]
    clusters = res[np.random.choice(rows, k, replace=False)]

    distance = np.zeros((rows, k))
    last = np.zeros((rows,))

    while True:
        for i in range(k):
            distance[:, i] = 1 - iou(res, clusters[i])
        current = np.argmin(distance, axis=1)

        if all(current == last):
            break

        last = current
        for i in range(k):
            clusters[i, :] = np.median(res[last == i], axis=0)

    avg_iou = np.mean(np.max(1-distance, axis=1))
    anchors = clusters.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])
    print(anchors, '\navg_iou=', avg_iou)
    return anchors


if __name__ == '__main__':
    # read_train_val()
    # read_test()
    anchors = anchor_box_k_mean(416, 9)

    with open('anchors.txt', 'w') as f:
        for [w, h] in anchors:
            f.write(''.join([str(w), ' ', str(h), '\r\n']))









