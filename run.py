from __future__ import print_function

import cv2
import sys
from time import time
import os

import kcftracker

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

duration = 0.01
duration_smooth = 0.01
detection_period = 20


def parse_label(label_line, data_format=''):
    if data_format == 'KITTI':
        val = label_line.split(' ')
        d = {'frame': int(val[0]), 'id': int(val[1]), 'type': val[2],
             'x1': int(float(val[6])), 'y1': int(float(val[7])), 'x2': int(float(val[8])), 'y2': int(float(val[9]))}
    elif data_format == 'VTB':
        val = label_line.split('\t')
        d = {'frame': int(val[0]) - 1, 'id': int(val[1]), 'type': val[2],
             'x1': int(float(val[3])), 'y1': int(float(val[4])),
             'x2': int(float(val[3]) + float(val[5])), 'y2': int(float(val[4]) + float(val[6]))}
    else:
        val = label_line.split('\t')
        d = {'frame': int(val[0].lstrip('0')) - 1, 'id': int(val[1]), 'type': val[2],
             'x1': int(float(val[3])), 'y1': int(float(val[4])), 'x2': int(float(val[5])), 'y2': int(float(val[6]))}

    return d


if __name__ == '__main__':

    # ============   Usage: run.py <filename> <det_result>   ============ #

    if len(sys.argv) == 1:
        sys.argv.append('C:/Users/Cory/Project/vid/videos/vid01.mp4')
        sys.argv.append('C:/Users/Cory/Project/vid/det/vid01_det.txt')

    assert len(sys.argv) == 3

    if not os.path.exists('output'):
        os.mkdir('output')

    cap = cv2.VideoCapture(sys.argv[1])
    labels_file = sys.argv[2]
    frames = list()
    for _ in range(int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))):
        frames.append(list())
    with open(labels_file) as labels:
        for line in labels.readlines():
            info = parse_label(line, 'vid')
            if info.get('id') != -1:  # pass unknown objects
                fi = info.get('frame')
                frames[fi].append(info)
    trackers = dict()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)) - 1  # current_frame = get next_frame - 1
        print('frame %d' % current_frame, end='')

        if current_frame % detection_period == 0:
            initTracking = True
        else:
            initTracking = False

        if initTracking:
            trackers.clear()
            all_targets = sorted(frames[current_frame], key=lambda d: d.get('id'))
            print(' ---------------- # trackers = %d' % len(all_targets), end='')
            for target in all_targets:
                # print(target)
                ix = target.get('x1')
                iy = target.get('y1')
                w = target.get('x2') - ix
                h = target.get('y2') - iy
                tid = target.get('id')

                trackers.update({tid: kcftracker.KCFTracker(True, False, True)})  # hog, fixed_window, multi-scale
                # if you use hog feature, there will be a short pause after you draw a first boundingbox,
                # that is due to the use of Numba.

                tracker = trackers.get(tid)
                tracker.init([ix, iy, w, h], frame)

            initTracking = False
            onTracking = True

        elif onTracking:
            t0 = time()
            for (tid, tracker) in trackers.items():
                # if tid in to_traced:
                (bbox, pv) = tracker.update(frame)
                bbox = map(int, bbox)
                if pv > 0.25:
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 1)
                    cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 1)
                else:
                    # cv2.rectangle(frame, (bbox[0], bbox[1]),
                    #              (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1)
                    cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 0), 1)
            t1 = time()
            duration = t1 - t0
            duration_smooth = 0.8 * duration_smooth + 0.2 * (t1 - t0)
            fps = 1 / duration_smooth
            print(' fsp = %4f' % fps, end='')
            cv2.putText(frame, 'FPS: ' + str(fps)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow('tracking', frame)
        cv2.imwrite('output/frame_%06d.jpg' % current_frame, frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break

        print()

    cap.release()
    cv2.destroyAllWindows()
