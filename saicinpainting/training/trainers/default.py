import logging
import torchvision
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from saicinpainting.training.data.datasets import make_constant_area_crop_params
from saicinpainting.training.losses.distance_weighting import make_mask_distance_weighter
from saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from saicinpainting.training.trainers.base import BaseInpaintingTrainingModule, make_multiscale_noise
from saicinpainting.utils import add_prefix_to_keys, get_ramp

LOGGER = logging.getLogger(__name__)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['GT'].shape[2],
                                                                             img_width=batch['GT'].shape[3],
                                                                             **kwargs)
    batch['GT'] = batch['GT'][:, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None, image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.A_Q_A = A_Q_A()
        # for param in self.A_Q_A.parameters():
        #     param.requires_grad = False
        # self.fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # for param in self.fasterrcnn.parameters():
        #     param.requires_grad = False
        # self.SR = SR()
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(**rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = make_mask_distance_weighter(**distance_weighter_kwargs) \
            if distance_weighter_kwargs is not None else None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr

        self.fake_fakes_proba = fake_fakes_proba
        if self.fake_fakes_proba > 1e-3:
            self.fake_fakes_gen = FakeFakesGenerator(**(fake_fakes_generator_kwargs or {}))

    def forward(self, batch):
        if self.training and self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)
        img = batch['input_image']
        # GT = batch['GT']
        mask = batch['mask']
        mask_img = torch.zeros(img.shape)
        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                img = img + (mask.unsqueeze(dim=1) - mask_img.unsqueeze(dim=1)) * noise[:, :img.shape[1]]
            img = torch.cat([img, noise], dim=1)
        if self.concat_mask:
            mask_img = torch.cat([img, mask], dim=1)
        batch['predicted_image'] = self.generator(mask_img)
        # mask_input = torch.zeros(1, 1, mask_img.shape[2], mask_img.shape[3]).to("cuda")
        # mask_input[:, batch['boarders'][0][0]:batch['boarders'][1][0], batch['boarders'][2][0]:batch['boarders'][3][0]] = 1
        # rev_input = torch.cat((batch['predicted_image'], mask_input), dim=1)
        # batch['rev_predicted_image'] = self.generator_rev(rev_input)
        # predicted_img_1024 = torch.zeros((mask_img.shape[0], 3, 512, 512))
        # predicted_img_1024[:, :, 128:512-128,  128:512-128] = batch["predicted_image"]
        # batch["predicted_image_1024"] = predicted_img_1024
        # batch['predicted_image'] = self.SR(batch['predicted_image'])
        if self.fake_fakes_proba > 1e-3:
            if self.training and torch.rand(1).item() < self.fake_fakes_proba:
                batch['fake_fakes'], batch['fake_fakes_masks'] = self.fake_fakes_gen(img, mask)
                batch['use_fake_fakes'] = True
            else:
                batch['fake_fakes'] = torch.zeros_like(img)
                batch['fake_fakes_masks'] = torch.zeros_like(mask)
                batch['use_fake_fakes'] = False

        batch['mask_for_losses'] = self.refine_mask_for_losses(img, batch['predicted_image'], mask) \
            if self.refine_mask_for_losses is not None and self.training \
            else mask

        return batch

    def generator_loss(self, batch):
        GT = batch['GT']
        predicted_img = batch["predicted_image"]
        input_image = batch['input_image']
        # rev_predicted_img = batch["rev_predicted_image"]
        # self.fasterrcnn.eval()
        # with torch.no_grad():
        #     rcnn_GT = self.fasterrcnn(GT)
        #     rcnn_pred = self.fasterrcnn(predicted_img)
        # rcnn_GT_inx = (rcnn_GT[0]['labels'] == 16).nonzero(as_tuple=True)[0]
        # rcnn_pred_inx = (rcnn_pred[0]['labels'] == 16).nonzero(as_tuple=True)[0]
        # if rcnn_pred_inx.shape[0] != 0 and rcnn_GT_inx.shape[0] != 0:
        #     rcnn_GT_cord = rcnn_GT[0]['boxes'][rcnn_GT_inx[0]]
        #     rcnn_pred_cord = rcnn_pred[0]['boxes'][rcnn_pred_inx[0]]
        #     bb_object_GT = GT[:, :, int(rcnn_GT_cord[1]):int(rcnn_GT_cord[3]), int(rcnn_GT_cord[0]):int(rcnn_GT_cord[2])]
        #     cx_pred = (int(rcnn_pred_cord[3]) + int(rcnn_pred_cord[1])) // 2
        #     cy_pred = (int(rcnn_pred_cord[2]) + int(rcnn_pred_cord[0])) // 2
        #     height_var = (int(rcnn_GT_cord[3]) - int(rcnn_GT_cord[1])) // 2
        #     width_var = (int(rcnn_GT_cord[2]) - int(rcnn_GT_cord[0])) // 2
        #
        #     if (cx_pred-height_var) >=0 and (cx_pred + height_var)<256 and (cy_pred - width_var)>=0 and (cy_pred + width_var) < 256:
        #         bb_object_pred = predicted_img[:, :, (cx_pred - height_var):(cx_pred + height_var), (cy_pred - width_var): (cy_pred + width_var)]
        #         bb_object_pred = torch.nn.functional.interpolate(bb_object_pred, size=(bb_object_GT.shape[2], bb_object_GT.shape[3]))
        #     else:
        #         bb_object_pred = predicted_img[:, :, int(rcnn_pred_cord[1]):int(rcnn_pred_cord[3]), int(rcnn_pred_cord[0]):int(rcnn_pred_cord[2])]
        #         bb_object_pred = torch.nn.functional.interpolate(bb_object_pred, size=(bb_object_GT.shape[2], bb_object_GT.shape[3]))
        #
        #     l1_value_for_object = masked_l1_loss(bb_object_pred, bb_object_GT,
        #                               self.config.losses.l1.weight_known,
        #                               self.config.losses.l1.weight_missing)
        # else:
        #     l1_value_for_object = 0
        #
        # total_loss = l1_value_for_object
        # L1
        l1_value = masked_l1_loss(predicted_img, GT,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, GT).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        self.adversarial_loss.pre_generator_step(real_batch=GT, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(GT)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=GT,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses.feature_matching.weight > 0:
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, GT)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value


        # for i in range(predicted_img.shape[0]):
        #     predicted_img_o = predicted_img[i]
        #     predicted_img_o = predicted_img_o[:, batch['boarders'][0][i]:batch['boarders'][1][i], batch['boarders'][2][i]:batch['boarders'][3][i]]
        #     predicted_img_o = torch.nn.functional.interpolate(predicted_img_o.unsqueeze(dim=0), size=(256, 256))[0]
        #     predicted_img_o = predicted_img_o[:, 16: 224+16, 16:224+16]
        #     if i==0:
        #         predicted_img_o_all = predicted_img_o.unsqueeze(dim=0)
        #     else:
        #         predicted_img_o_all = torch.cat((predicted_img_o_all, predicted_img_o.unsqueeze(dim=0)), dim=0)
        # score = self.A_Q_A(predicted_img_o_all)
        # self.loss_score = (1 / (score + 0.01)).mean()
        # metrics['loss_aesthetic'] = self.loss_score
        # total_loss += 20 * self.loss_score
        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach().contiguous()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['GT'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['GT'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['GT'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred)
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['GT'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        return total_loss, metrics
