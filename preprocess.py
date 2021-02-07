import os


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
                f.write(''.join([ann, '/', name, '.xml', ' ',
                                 img, '/', name, '.jpg', '\r\n']))


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
            f.write(''.join([ann_path, '/', name, '.xml', ' ',
                             img_path, '/', name, '.jpg', '\r\n']))


if __name__ == '__main__':
    read_train_val()
    read_test()








