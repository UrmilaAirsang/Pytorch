import cv2
import torch
import numpy as np

from skimage.measure import find_contours


from .visual_utils import ColorPalette, apply_mask

def run(model, image_path, size, thresh=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_read = cv2.imread(image_path)
    image_resized = cv2.resize(image_read, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).unsqueeze(0).to(device).float() / 255.0

    model.eval()
    with torch.no_grad():
        _, req, _, _ = model(image)
        req = torch.sigmoid(req)

    class_ids = list(range(8))
    masks = np.transpose(req.cpu().numpy()[0], (1, 2, 0))
    masks[masks > thresh] = 1
    masks[masks <= thresh] = 0

    N = len(class_ids)
    instance_colors = ColorPalette(N).getColorMap(returnTuples=True)
    class_colors = ColorPalette(11).getColorMap(returnTuples=True)
    class_colors[0] = (128, 128, 128)

    masked_image = image_resized.astype(np.uint8).copy()

    for i in range(N):
        ## Label
        class_id = class_ids[i]

        ## Mask
        mask = masks[:, :, i]

        masked_image = apply_mask(masked_image.astype(np.float32), mask, instance_colors[i]).astype(np.uint8)

        ## Mask Polygon
        ## Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            ## Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            cv2.polylines(masked_image, np.expand_dims(verts.astype(np.int32), 0), True,
                          color=class_colors[int(class_id)])
            continue

        continue

    return masked_image, masks