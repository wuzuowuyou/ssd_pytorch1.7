# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp

# criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
#                          False, args.cuda)

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).

                loc_data [3,8732,4]
                conf_data [3,8732,21]
                priors [8732,4]
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4) #[3,8732,4]
        conf_t = torch.LongTensor(num, num_priors) #[3,8732]
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False) #[3,8732,4]
        conf_t = Variable(conf_t, requires_grad=False) #[3,8732]

        pos = conf_t > 0 #pos [3,8732]  False,True
        num_pos = pos.sum(dim=1, keepdim=True) #num_pos shape[3,1]   | [23],[4],[9]

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]  #pos.unsqueeze(pos.dim()) [3,8732,1]
        #pos_idx [3,8732,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        #loc_data[3, 8732, 4]   aa[240]
        aa = loc_data[pos_idx]
        #[3,8732,4]   bb[240]
        bb = loc_t[pos_idx]

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        #loss_l tensor(14.0165, grad_fn=<SmoothL1LossBackward>)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        #conf_data [3,8732,21]  batch_conf[3*8732,21]  [26196,21]
        batch_conf = conf_data.view(-1, self.num_classes)
        b1 = log_sum_exp(batch_conf) #[26196,1]
        b2 = batch_conf.gather(1, conf_t.view(-1, 1)) #[26196,1]

        #loss_c[26196,1]    #https://zhuanlan.zhihu.com/p/153535799
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        #loss_c[pos] = 0  # filter out pos boxes for now
        #loss_c = loss_c.view(num, -1)
        #loss_c [3,8732]
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0 #把正样本的loss置为0
        #loss_idx [3,8732]
        tmp1, loss_idx = loss_c.sort(1, descending=True) ## _, loss_idx = loss_c.sort(1, descending=True)
        #idx_rank [3,8732]
        tmp2, idx_rank = loss_idx.sort(1) ## _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)#num_pos shape[3,1]   | [23],[4],[9]
        aaaaa = self.negpos_ratio * num_pos
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)#num_pos shape[3,1]   | [69],[12],[27]
        #neg [3,8732]  True,False  给出的是conf_data对应坐标的True与False 排序的从大到小
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)#pos[3,8732]  conf_data[3,8732,21]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)##neg [3,8732]  conf_data[3,8732,21]
        ## pos_idx+neg_idx  这两者的形状都是相同的[3,8732,21] 值都是True或者False  加运算相当执行了或运算，只要有一个True就是True
        #conf_p [144,21] -->  这里面的144就是上面两个pos_idx和neg_idx里面True数量之和 69+12+27+23+4+9=144
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # pos [3,8732]
        # neg [3,8732]
        # conf_t [3,8732]
        # targets_weighted [144]
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #loss_c tensor(58.0656, grad_fn=<NllLossBackward>)
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum() ##N=36  就是num_pos之和[23] + [4] + [9]
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
