import numpy as np
import scipy.misc as misc
import os.path as path
import glob

class DatasetReader():
    def __init__(self, imageset_index, resize=[224, 224], isShuffle=True):
        print("Initialize Dataset Reader ...")

        # self.index_file = path.join(imageset_dir, 'index.txt')
        self.index_file = path.join('./data/data3500/', imageset_index)
        # self.img_files, self.ant_files = read_index(self.index_file, imageset_dir)
        self.img_files, self.ant_files = read_index(self.index_file, './data/data3500/')

        self.isShuffle = isShuffle

        if resize == []:
            self.resize = False
        else:
            self.resize = True
            self.height = resize[0]
            self.width = resize[1]

        self.num = len(self.img_files)
        # 这里全部读取真的好吗，一定要这样写吗
        self.imgs = self._read_images(self.img_files) # 这里全部读取所有的图片是不是就是显存报错的原因?

        self.ants = self._read_images(self.ant_files)

        # initialize batch offset and epoch
        self.reset_batch_offset()
        self.reset_epoch_count()

    def reset_batch_offset(self):
        self.batch_offset = 0

    def reset_epoch_count(self):
        self.epoch_count = 0

    def _read_images(self, image_files):
        # return np.array([self._read_image(img_file) if path.exists(img_file) else print(img_file) for img_file in image_files])
        # 为了加一个读取的进度显示，所以改写复杂形式
        ans =[]
        n = len(image_files)
        for img_file in image_files:
            if path.exists(img_file):
                ans.append(self._read_image(img_file))
                print('\r',' _read_images:%d/%d,%s' % (image_files.index(img_file),n,img_file), end='')
            else:
                print(img_file)
        return np.array(ans)

    def _read_image(self, image_file):
        # print('>>> _read_image: ',image_file)###
        image = misc.imread(image_file)
        if self.resize:
            # resize_image = misc.imresize(image, [self.width, self.height], interp='nearest')
            resize_image = misc.imresize(image, [self.height, self.width], interp='nearest')
        else:
            resize_image = image
        # print('>>> _read_image',image.shape,resize_image.shape,self.resize)
        # expand 3-dimension tensor to 4-dimension tensor
        if len(resize_image.shape) == 2:
            resize_image = np.expand_dims(resize_image, axis=2)

        # check fate jpg - rgb
        if image_file[-3:] == 'jpg':
            # 这里图片尺寸的assert有点粗暴了，故意的？
            # assert resize_image.shape == (224, 224, 3), print(image_file)
            # print(resize_image.shape)
            assert resize_image.shape == (128, 1024, 3), print(image_file)

        # check fate jpg - gray
        if image_file[-3:] == 'png':
            assert resize_image.shape == (128, 1024, 1), print(image_file)
            resize_image = np.divide(resize_image, 255).astype(int) # 这里把标签图片变成0，1了

        return resize_image

    def next_batch(self, batch_size):
        start = self.batch_offset
        end = start + batch_size
        if end <= self.num:
            self.batch_offset = end
            return self.imgs[start: end], self.ants[start: end]
        else:
            # finish one epoch and reset the batch offset
            self.epoch_count += 1
            self.reset_batch_offset()
            # when an epoch finishes, the sequence is reset
            if self.isShuffle:
                sequence = np.arange(self.num)
                np.random.shuffle(sequence)
                self.imgs = self.imgs[sequence]
                self.ants = self.ants[sequence]

            return self.next_batch(batch_size)
# 预测的时候用
class ImageReader():
    def __init__(self, image_dir):
        self.img_files = glob.glob(path.join(image_dir, '*.jpg'))
        self.save_names = [img_file.replace(".jpg", ".png") for img_file in self.img_files]
        self.num = len(self.img_files)

        self.img_index = 0

    def _read_image(self, image_file):
        image = misc.imread(image_file, mode='RGB')
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        return image

    def next_image(self):
        if self.img_index < self.num:
            image = self._read_image(self.img_files[self.img_index])
            name = self.save_names[self.img_index]
            shape = image.shape
            self.img_index += 1
        else:
            self.img_index = 0
            image, name, shape = self.next_image()
        return image, name, shape[:2]

def read_index(index_file, dataset_dir):

    image_files = []
    annotation_files = []
    with open(index_file, 'r') as file:
        for row in file.readlines():
            image_file, annotation_file = row[:-1].split(',')

            image_files.append(dataset_dir + '/' + image_file)
            annotation_files.append(dataset_dir + '/' + annotation_file)
    return image_files, annotation_files

if __name__ == '__main__':
    # datasetReader = DatasetReader('data/valid')
    # for i in range(60):
    #     a, b = datasetReader.next_batch(10)
    #     print(datasetReader.epoch_count, datasetReader.batch_offset)
    #     print(a.shape, b.shape)
    imagedata = ImageReader('compare_cracknet')
    for i in range(imagedata.num):
        print(imagedata.next_image())
