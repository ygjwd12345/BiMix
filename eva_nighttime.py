import os
import torch
import numpy as np

from PIL import Image
import torch.nn as nn
from torch.utils import data

from network import *
from dataset.nighttime_dataset import nigthttimedataset

from configs.test_nighttime_config import get_arguments
from compute_iou import compute_mIoU_nighttime

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def main(j):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")


    args = get_arguments()
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes)
    if args.model == 'RefineNet':
        model = RefineNet(num_classes=args.num_classes, imagenet=False)
    ### set path
    args.restore_from='./snapshots/RefineNet/dannet'+str(j)+'.pth'
    args.restore_from_light='./snapshots/RefineNet/dannet_light'+str(j)+'.pth'
    # print(args.restore_from)
    saved_state_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    # lightnet = LightNet()
    # lightnet = mlpgan()
    lightnet =enhance_net_nopool()
    saved_state_dict = torch.load(args.restore_from_light)
    model_dict = lightnet.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    lightnet.load_state_dict(saved_state_dict)

    model = model.to(device)
    lightnet = lightnet.to(device)
    model.eval()
    lightnet.eval()
    testloader = data.DataLoader(nigthttimedataset(args.data_dir, args.data_list, set=args.set))
    interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

    weights = torch.log(torch.FloatTensor(
        [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
         0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
         0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0

    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print('%d processd' % index)
        image, _,_,name = batch
        image = image.to(device)
        with torch.no_grad():
            # r= lightnet(image)

            _,r,_ = lightnet(image)
            enhancement = image + r
            # enhancement = image
            if args.model == 'RefineNet':
                output2 = model(enhancement)
            else:
                _, output2 = model(enhancement)

        weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)
        output2 = output2 * weights_prob
        output = interp(output2).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ###### get the enhanced image
        enhancement = enhancement.cpu().data[0].numpy().transpose(1,2,0)
        enhancement = (enhancement/2)*mean_std[1]+mean_std[0]
        enhancement = (enhancement-enhancement.min())/(enhancement.max()-enhancement.min())
        enhancement = enhancement[:, :, ::-1]*255  # change to BGR
        enhancement = Image.fromarray(enhancement.astype(np.uint8))

        ###### get the light
        # light = r.cpu().data[0].numpy().transpose(1,2,0)
        # light = (light-light.min())/(light.max()-light.min())
        # light = light[:, :, ::-1]*255  # change to BGR
        # light = Image.fromarray(light.astype(np.uint8))


        name = name[0].split('/')[-1]
        # output.save('%s/%s' % (args.save, name))
        # output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))
        enhancement.save('%s/%s_enhancement.png' % (args.save, name.split('.')[0]))
        # light.save('%s/%s_light.png' % (args.save, name.split('.')[0]))


if __name__ == '__main__':
    # f = open("result_driving.txt", "w")
    ### for all
    # for i in range(99):
    #     j=(i+1)*1000
    #     main(j)
    #     mIou=compute_mIoU_nighttime('./dataset/NighttimeDrivingTest', './result/dannet_RefineNet_driving', './dataset/lists')
    #     f.write('Ep:%dK mIoU:%s\r\n'%(i+1,str(round(np.nanmean(mIou) * 100, 2))))
    ### for pick up
    i = 83
    j = (i + 1) * 1000
    main(j)
    # mIou = compute_mIoU_nighttime('./dataset/NighttimeDrivingTest', './result/dannet_RefineNet_driving', './dataset/lists')
    # f.write('Ep:%dK mIoU:%s\r\n' % (i + 1, str(round(np.nanmean(mIou) * 100, 2))))