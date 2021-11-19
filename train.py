import os

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

from network import *


from dataset.zurich_pair_dataset import zurich_pair_DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from configs.train_config import get_arguments
## class mix
from utils import transformmasks
from utils import transformsgpu
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
import random

### vis log set
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

withwandb = False
try:
    import wandb
except ImportError:
    withwandb = False
    print('WandB disabled')

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs() ** 2)


def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    data, target = transformsgpu.oneMix(mask = parameters["Mix"], data = data, target = target)
    # data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    # data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target
def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters
def main():
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")

    cudnn.enabled = True
    cudnn.benchmark = True

    if withwandb:
        wandb.init(project="n2d")

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes)
    if args.model == 'RefineNet':
        model = RefineNet(num_classes=args.num_classes, imagenet=False)
    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    model.train()
    model.to(device)

    # lightnet = LightNet()
    lightnet =enhance_net_nopool()

    lightnet.train()
    lightnet.to(device)

    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1.train()
    model_D1.to(device)

    model_D2 = FCDiscriminator(num_classes=args.num_classes)
    model_D2.train()
    model_D2.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        cityscapesDataSet(args, args.data_dir, args.data_list,
                          max_iters=args.num_steps * args.iter_size * args.batch_size,
                          set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(zurich_pair_DataSet(args, args.data_dir_target, args.data_list_target,
                                                       max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                       set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(list(model.parameters()) + list(lightnet.parameters()),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.zero_grad()
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                           0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                           0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                           0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    static_loss = StaticLoss(num_classes=11, weight=weights[:11])
    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()
    loss_idt= nn.L1Loss()
    ### class mix
    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=255).cuda()


    interp = nn.Upsample(size=(args.input_size, args.input_size), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.input_size_target, args.input_size_target), mode='bilinear',
                                align_corners=True)

    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):
        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_pseudo = 0
        loss_D_value1 = 0
        loss_D_value2 = 0
        loss_d2n=0
        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)
        optimizer_D1.zero_grad()
        adjust_learning_rate_D(args, optimizer_D1, i_iter)
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(args, optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):
            # load data
            _, batch = targetloader_iter.__next__()
            images_n, images_d, _, _ = batch
            images_d = images_d.to(device)
            images_n = images_n.to(device)

            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)
            # train G
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False

            # train with target
            mean_light = images_n.mean()
            # r= lightnet(images_d)
            _,r,A= lightnet(images_d)

            enhanced_images_d = images_d + r
            loss_enhance = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images_d, images_d)) \
                           + torch.mean(loss_exp_z(enhanced_images_d, mean_light))

            if args.model == 'RefineNet':
                pred_target_d = model(enhanced_images_d)
            else:
                _, pred_target_d = model(enhanced_images_d)
            pred_target_d = interp(pred_target_d)
            pred_target_d_ = interp_target(pred_target_d)

            ### classmix with  day night image
            weak_parameters = {"flip": 0}
            image_T_d, _ = weakTransform(weak_parameters, data=images_d)
            if args.model == 'RefineNet':
                logits_T_d = model(image_T_d)
            else:
                _, logits_T_d = model(image_T_d)
            logits_T_d = interp(logits_T_d)
            logits_T_d, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data=logits_T_d.detach())

            pseudo_label_d = torch.softmax(logits_T_d.detach(), dim=1)
            max_probs_d, pred_d = torch.max(pseudo_label_d, dim=1)
            for image_i in range(args.batch_size):
                classes = torch.unique(pred_d[image_i])
                ## dynamic pick up
                index1=30>classes
                index2=classes>11
                index=index1&index2
                classes = (classes[index]).cuda()
                if image_i == 0:
                    MixMask0_d2n = transformmasks.generate_class_mask(pred_d[image_i], classes).unsqueeze(0).cuda()
                else:
                    MixMask1_d2n = transformmasks.generate_class_mask(pred_d[image_i], classes).unsqueeze(0).cuda()
            strong_parameters = {"Mix": MixMask0_d2n}

            strong_parameters["flip"] = 0
            strong_parameters["ColorJitter"] = random.uniform(0, 1)
            strong_parameters["GaussianBlur"] = random.uniform(0, 1)

            image_M_d2n0, _ = strongTransform(strong_parameters,
                                              data=torch.cat((images_d[0].unsqueeze(0), images_n[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1_d2n
            image_M_d2n1, _ = strongTransform(strong_parameters,
                                              data=torch.cat((images_d[1].unsqueeze(0), images_n[1].unsqueeze(0))))
            image_M_d2n = torch.cat((image_M_d2n0, image_M_d2n1))
            _, r_mix, _ = lightnet(image_M_d2n)
            # loss_d2n=loss_idt(r_mix[0]*MixMask0,r[0]*MixMask0_d2n)+loss_idt(r_mix[1]*MixMask1,r[1]*MixMask1_d2n)
            loss_d2n=loss_idt(r_mix,r)

            D_out_d = model_D1(F.softmax(pred_target_d_, dim=1))
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(source_label).to(device)
            loss_adv_target_d = weightedMSE(D_out_d, D_label_d)
            loss = 0.01 * loss_adv_target_d + 0.01 * loss_enhance+0.001*loss_d2n
            loss = loss / args.iter_size
            loss.backward()

            # r = lightnet(images_n)
            _,r,A= lightnet(images_n)

            enhanced_images_n = images_n + r
            loss_enhance = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images_n, images_n)) \
                           + torch.mean(loss_exp_z(enhanced_images_n, mean_light))

            if args.model == 'RefineNet':
                pred_target_n = model(enhanced_images_n)
            else:
                _, pred_target_n = model(enhanced_images_n)
            pred_target_n = interp(pred_target_n)
            pred_target_n_ = interp_target(pred_target_n)
            D_out_n_19 = model_D2(F.softmax(pred_target_n_, dim=1))
            D_label_n_19 = torch.FloatTensor(D_out_n_19.data.size()).fill_(source_label).to(device)
            loss_adv_target_n_19 = weightedMSE(D_out_n_19, D_label_n_19 )

            ### classmix with night image
            weak_parameters = {"flip": 0}
            image_T_n, _ = weakTransform(weak_parameters, data=enhanced_images_n)
            if args.model == 'RefineNet':
                logits_T_n = model(image_T_n)
            else:
                _, logits_T_n = model(image_T_n)
            logits_T_n = interp(logits_T_n)
            logits_T_n, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data=logits_T_n.detach())

            pseudo_label_n = torch.softmax(logits_T_n.detach(), dim=1)
            max_probs_n, pred_n = torch.max(pseudo_label_n, dim=1)

            for image_i in range(args.batch_size):
                classes = torch.unique(labels[image_i])
                # classes=classes[classes!=ignore_label]
                ## random pick up
                nclasses = classes.shape[0]
                if nclasses > 0:
                    classes = (classes[torch.Tensor(
                        np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).cuda()
                ## dynamic pick up
                # index1=30>classes
                # index2=classes>11
                # index=index1&index2
                # classes = (classes[index]).cuda()
                if image_i == 0:
                    MixMask0 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
                else:
                    MixMask1 = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
            strong_parameters = {"Mix": MixMask0}

            strong_parameters["flip"] = 0
            strong_parameters["ColorJitter"] = random.uniform(0, 1)
            strong_parameters["GaussianBlur"] = random.uniform(0, 1)

            image_M_s2n0, _ = strongTransform(strong_parameters,
                                              data=torch.cat((images[0].unsqueeze(0), images_n[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            image_M_s2n1, _ = strongTransform(strong_parameters,
                                              data=torch.cat((images[1].unsqueeze(0), images_n[1].unsqueeze(0))))
            image_M_s2n = torch.cat((image_M_s2n0, image_M_s2n1))
            if args.model == 'RefineNet':
                logits_M_s2n = model(image_M_s2n)
            else:
                _, logits_M_s2n = model(image_M_s2n)
            logits_M_s2n = interp(logits_M_s2n)
            if i_iter % args.save_pred_every == 0 and i_iter != 0:
                if withwandb:
                    jet = plt.get_cmap('binary')
                    cNorm = colors.Normalize(vmin=0, vmax=1)
                    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
                    # colorVal = scalarMap.to_rgba(MixMask0.cpu().squeeze())
                    # wandb.log({"s2n1": [wandb.Image(images[0], caption="cityscape"),wandb.Image(images_n[0], caption="night"),wandb.Image(image_M_s2n0, caption="mix"),wandb.Image(colorVal, caption="mask")]})
                    colorVal = scalarMap.to_rgba(MixMask0_d2n.cpu().squeeze())
                    wandb.log({"d2n1": [wandb.Image(images_d[0], caption="day"),wandb.Image(images_n[0], caption="night"),wandb.Image(image_M_d2n[0], caption="mix"),wandb.Image(colorVal, caption="mask")]})
                    # colorVal = scalarMap.to_rgba(MixMask1.cpu().squeeze())
                    # wandb.log({"s2n2": [wandb.Image(images[1], caption="cityscape"),wandb.Image(images_n[1], caption="night"),wandb.Image(image_M_s2n1, caption="mix"),wandb.Image(colorVal, caption="mask")]})
                    colorVal = scalarMap.to_rgba(MixMask1_d2n.cpu().squeeze())
                    wandb.log({"d2n2": [wandb.Image(images_d[1], caption="day"),wandb.Image(images_n[1], caption="night"),wandb.Image(image_M_d2n[1], caption="mix"),wandb.Image(colorVal, caption="mask")]})
                    wandb.log({"lightnet": [wandb.Image(r_mix[0], caption="re1"),wandb.Image(image_M_d2n[0], caption="image1"),wandb.Image(r_mix[1], caption="re2"),wandb.Image(image_M_d2n[1], caption="image2")]})

            psudo_prob = torch.zeros_like(pred_target_d)
            threshold = torch.ones_like(pred_target_d[:, :11, :, :]) * 0.2
            threshold[pred_target_d[:, :11, :, :] > 0.4] = 0.8
            psudo_prob[:, :11, :, :] = threshold * pred_target_d[:, :11, :, :].detach() + (
                        1 - threshold) * pred_target_n[:, :11, :, :].detach()
            psudo_prob[:, 11:, :, :] = pred_target_n[:, 11:, :, :].detach()

            weights_prob = weights.expand(psudo_prob.size()[0], psudo_prob.size()[3], psudo_prob.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            psudo_prob = psudo_prob * weights_prob
            psudo_gt = torch.argmax(psudo_prob.detach(), dim=1)
            psudo_gt_=psudo_gt
            psudo_gt_[psudo_gt >= 11] = 255
            loss_pseudo = static_loss(logits_M_s2n[:, :11, :, :], psudo_gt_.detach())

            strong_parameters["Mix"] = MixMask0
            _, pred_M_s2n0 = strongTransform(strong_parameters, target = torch.cat((labels[0].unsqueeze(0),psudo_gt[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, pred_M_s2n1 = strongTransform(strong_parameters, target = torch.cat((labels[1].unsqueeze(0),psudo_gt[1].unsqueeze(0))))
            psudo_gt = torch.cat((pred_M_s2n0,pred_M_s2n1)).long()
            ### generate pixel wise weight
            unlabeled_weight = torch.sum(max_probs_n.ge(0.968).long() == 1).item() / np.size(np.array(psudo_gt.cpu()))
            pixelWiseWeight = unlabeled_weight * torch.ones(max_probs_n.shape).cuda()
            onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
            strong_parameters["Mix"] = MixMask0
            _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
            strong_parameters["Mix"] = MixMask1
            _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
            pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()
            loss_s2n=unlabeled_loss(logits_M_s2n, psudo_gt.detach(), pixelWiseWeight.detach())

            loss = 0.01 * loss_adv_target_n_19+ 0.01 * loss_enhance+loss_pseudo+0.001 *loss_s2n
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value += loss_adv_target_n_19.item() / args.iter_size

            # train with source

            # r=lightnet(images)
            _,r,A= lightnet(images)

            enhanced_images = images + r
            loss_enhance = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images, images)) \
                           + torch.mean(loss_exp_z(enhanced_images, mean_light))

            if args.model == 'RefineNet':
                pred_c = model(enhanced_images)
            else:
                _, pred_c = model(enhanced_images)

            pred_c = interp(pred_c)
            loss_seg = seg_loss(pred_c, labels)
            loss = loss_seg + loss_enhance
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.item() / args.iter_size

            # train D
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred_c = pred_c.detach()
            D_out1 = model_D1(F.softmax(pred_c, dim=1))
            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value2 += loss_D1.item()

            pred_c = pred_c.detach()
            D_out2 = model_D2(F.softmax(pred_c, dim=1))
            D_label2 = torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / args.iter_size / 2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target_d = pred_target_d.detach()
            D_out1 = model_D1(F.softmax(pred_target_d, dim=1))
            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            pred_target_n = pred_target_n.detach()
            D_out2 = model_D2(F.softmax(pred_target_n, dim=1))
            D_label2 = torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / args.iter_size / 2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print(
            'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv = {3:.3f}, loss_D1 = {4:.3f}, loss_D2 = {5:.3f}, loss_pseudo = {6:.3f},loss_consistancy={7:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value,
                loss_adv_target_value, loss_D_value1, loss_D_value2, loss_pseudo,loss_d2n))
        if withwandb:
            wandb.log(
                {"loss_seg": loss_seg_value, "loss_adv": loss_adv_target_value, "loss_D1": loss_D_value1, "loss_D2": loss_D_value2, "loss_pseudo": loss_pseudo,'loss_s2n':loss_s2n,'loss_d2n':loss_d2n})

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'dannet' + str(i_iter) + '.pth'))
            torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, 'dannet_light' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d1_' + str(i_iter) + '.pth'))
            torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d2_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()