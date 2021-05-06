def freeze(model, base=True, yolo=False, depth_decoder=False, plane_decoder=False):
    to_freeze = []
    if base:
        to_freeze.append('pretrained')
    if yolo:
        to_freeze.append('yolo')
    if depth_decoder:
        to_freeze.append('scratch')
    if plane_decoder:
        to_freeze.append('plane_segmentation_decode')

    for k, v in dict(model.named_parameters()).items():
        for freeze_key in to_freeze:
            if freeze_key in k:
                v.requires_grad = False


def unfreeze(model, base=True, yolo=True, depth_decoder=True, plane_decoder=True):
    to_unfreeze = []
    if base:
        to_unfreeze.append('pretrained')
    if yolo:
        to_unfreeze.append('yolo')
    if depth_decoder:
        to_unfreeze.append('scratch')
    if plane_decoder:
        to_unfreeze.append('plane_segmentation_decode')

    for k, v in dict(model.named_parameters()).items():
        for freeze_key in to_unfreeze:
            if freeze_key in k:
                v.requires_grad = True