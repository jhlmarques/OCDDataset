_base_ = [
    './rotated-retinanet-rbox-le90_r50_fpn_rr_1x_ocd-cropped.py'
]

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='probiou', loss_weight=5.0)))
