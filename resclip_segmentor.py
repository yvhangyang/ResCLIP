import logging
import sys

import torch
import torch.nn as nn
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS

import clip
from pamr import PAMR
from prompts.imagenet_template import openai_imagenet_template

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")


@MODELS.register_module()
class ResCLIPSegmentation(BaseSegmentor):
    def __init__(self, clip_path, name_path, device=torch.device('cuda'),
                 arch='wo_resi', attn_strategy='resclip', gaussian_std=5., pamr_steps=0, use_rcs=True, use_sfr=True,
                 pamr_stride=(8, 16), prob_thd=0.0, area_thd=None, logit_scale=40, slide_stride=112,
                 slide_crop=224, temp_thd=None, attn_sfr_weights=None, attn_rcs_weights=None, delete_same_entity=False):

        data_preprocessor = SegDataPreProcessor(mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323], rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)
        self.net, _ = clip.load(clip_path, device=device, jit=False)

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = list()
        with torch.no_grad():
            for qw in query_words:
                query = clip.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.net.visual.set_params(arch, attn_strategy, gaussian_std)
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False
        self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device) if pamr_steps > 0 else None

        self.temp_thd = temp_thd
        self.attn_sfr_weights = attn_sfr_weights
        self.attn_rcs_weights = attn_rcs_weights
        self.delete_same_entity = delete_same_entity
        self.use_rcs=use_rcs,
        self.use_sfr=use_sfr,



    def forward_feature(self, img):
        if type(img) == list:
            img = img[0]

        # (1,197,512)
        image_features = self.net.encode_image(img, return_all=True, use_rcs=self.use_rcs, use_sfr=self.use_sfr,
                                               temp_thd=self.temp_thd, attn_sfr_weights=self.attn_sfr_weights,
                                               attn_rcs_weights=self.attn_rcs_weights, delete_same_entity=self.delete_same_entity,
                                               query_features=self.query_features, img=img,
                                               patch_size=self.net.visual.patch_size, logit_scale=self.logit_scale, dtype=self.dtype,
                                               align_corners=self.align_corners)
        image_features = image_features[:, 1:]
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ self.query_features.T  # (1,196,512)*(150,512).T

        patch_size = self.net.visual.patch_size
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        # (1,150,14,14)->(1,150,224,224)
        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear', align_corners=self.align_corners)

        return logits

    def forward_slide(self, img, stride=112, crop_size=224):
        """
        Inference by sliding-window with overlap. If h_crop > h_img or w_crop > w_img,
        the small patch will be used to decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]  # (1,3,224,224)

                crop_seg_logit = self.forward_feature(crop_img)  # (1,150,224,224)

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        logits = preds / count_mat  # (1, 150, 336, 448)

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        # seg_logits:(1, 150, 512, 683)  # inputs: (1,3,336,448)
        if self.slide_crop > 0:  # sliding windows
            seg_logits = self.forward_slide(inputs, self.slide_stride, self.slide_crop)

        else:  # if slide_crop=0
            # VOC20：(1,30,336,459) --> (1,30,336,459)
            seg_logits = self.forward_feature(inputs)

        # Both interpolate
        img_size = batch_img_metas[0]['ori_shape']
        # ade20K: seg_logits:[1,150,336,448] --> [1,150,512,683]
        seg_logits = nn.functional.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)

        if self.pamr:
            img = nn.functional.interpolate(inputs, size=img_size, mode='bilinear', align_corners=self.align_corners)
            seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)

        data_samples = self.postprocess_result(seg_logits, data_samples)
        return data_samples

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_probs = torch.softmax(seg_logits[i] * self.logit_scale, dim=0)  # n_queries * w * h
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_probs = seg_probs.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_probs = (seg_probs * cls_index).max(1)[0]

            if self.area_thd is not None:  # SCLIP setting
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)
                seg_logits[1:] *= area_pred.transpose(0, -1)

            seg_pred = seg_probs.argmax(0, keepdim=True)
            seg_pred[seg_probs.max(0, keepdim=True)[0] < self.prob_thd] = 0
            seg_probs /= seg_probs.sum(0, keepdim=True)

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits': PixelData(**{'data': seg_probs}),
                    'pred_sem_seg': PixelData(**{'data': seg_pred})
                })

        return data_samples

    def _forward(data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = list(), list()
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
