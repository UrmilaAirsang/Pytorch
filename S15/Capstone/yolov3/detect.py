import argparse
from sys import platform

from datasets import *
from .utils import *


def detect(data, model, img_size, device, source, out,
           save_img=False, conf_thres=0.001,
         iou_thres=0.6):
    with torch.no_grad():
        # Eval mode
        model.to(device).eval()

        # Fuse Conv2d + BatchNorm2d layers

        dataset = LoadImages(source, img_size=img_size)

        # Get names and colors
        names = load_classes(data['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        view_img = True

        # Run inference
        t0 = time.time()
        _ = model(torch.zeros((1, 3, img_size, img_size), device=device)) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            _, _, pred, _ = model(img)
            t2 = time_synchronized()



            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres,
                                       multi_label=False, classes=names)


            # Process detections
            for i, det in enumerate(pred):  # detections per image

                p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)

        print('Done. (%.3fs)' % (time.time() - t0))





