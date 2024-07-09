_base_ = './rotated-retinanet-rbox-le90_r50_fpn_rr_1x_ocd-cropped.py'

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=5.5)))
