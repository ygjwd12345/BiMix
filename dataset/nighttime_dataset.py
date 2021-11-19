import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from dataset.transforms import *
import torchvision.transforms as standard_transforms


class nigthttimedataset(data.Dataset):
    def __init__(self,root, list_path, max_iters=None, set='test'):
        self.root = root
        self.list_path = list_path

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        val_input_transform = standard_transforms.Compose([
            standard_transforms.Resize((540, 960)),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        self.transform = val_input_transform
        self.target_transform = extended_transforms.MaskToTensor()

        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s/%s" % (self.set,'night', name))
            label_file = osp.join(self.root, "gtCoarse_daytime_trainvaltest/%s/%s/%s" % (self.set,'night', name)).replace("leftImg8bit", "gtCoarse_labelIds")
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        label = np.asarray(label, np.uint8)

        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        label = Image.fromarray(label_copy.astype(np.uint8))

        label = self.target_transform(label)

        image = self.transform(image)

        size = image.shape
        return image, label, np.array(size), name


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    import os
    # from configs.test_config import get_arguments
    # args = get_arguments()

    local_path = "/data2/gyang/DANNet/dataset/NighttimeDrivingTest/leftImg8bit/test/night"
    files = os.listdir(local_path)
    file = open('label_nighttime.txt', 'w+')
    for i in files:
        a="gtCoarse_daytime_trainvaltest/%s/%s/%s" % ('test', 'night', i)

        file.write(a.replace("leftImg8bit", "gtCoarse_labelIds") + '\n')
    file.close()
    # dst = nigthttimedataset(args,local_path,)
    # bs = 4
    # trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    # for i, data in enumerate(trainloader):
    #     imgs, labels = data
    #     imgs = imgs.numpy()[:, ::-1, :, :]
    #     imgs = np.transpose(imgs, [0, 2, 3, 1])
    #     f, axarr = plt.subplots(bs, 2)
    #     for j in range(bs):
    #         axarr[j][0].imshow(imgs[j])
    #         axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    #     plt.show()
    #     plt.close()

