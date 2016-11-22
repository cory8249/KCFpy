from __future__ import print_function

import cv2
import sys
from time import time
import os
from multiprocessing import Queue

import kcftracker
from tracker_mp import TrackerMP
from config import *


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
        sys.argv.append(default_input_path)
        sys.argv.append(default_det_path)

    assert len(sys.argv) == 3
    
    if not os.path.exists('output'):
        os.mkdir('output')
    
    input_v_path = sys.argv[1]
    labels_file = sys.argv[2]
    if input_v_path.find('mp4') != -1:
        input_mode = 'video'
    else:
        input_mode = 'image'

    frames = list()
    if input_mode == 'video':
        cap = cv2.VideoCapture(input_v_path)
        assert cap.isOpened()
        frames_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        files_in_dir = sorted([f for f in os.listdir(input_v_path) if os.path.isfile(os.path.join(input_v_path, f))])
        frames_count = len(files_in_dir)
    
    for _ in range(frames_count):
        frames.append(list())
    with open(labels_file) as labels:
        for line in labels.readlines():
            info = parse_label(line, 'vid')
            if info.get('id') != -1:  # pass unknown objects
                fi = info.get('frame')
                frames[fi].append(info)
    trackers = dict()

    duration = 0.01
    duration_smooth = 0.01

    for current_frame in range(frames_count):
        if input_mode == 'video':
            ret, frame = cap.read()
        else:
            current_frame_path = input_v_path + '/%06d.jpg' % (current_frame + 1)
            frame = cv2.imread(current_frame_path)

        if frame is None:
            print('read image/video error at frame', current_frame)
            if input_mode == 'image':
                print(current_frame_path)
            raise IOError

        print('frame %d' % current_frame, end='')

        if current_frame % detection_period == 0:
            initTracking = True
        else:
            initTracking = False

        if initTracking:

            # clear old trackers
            for tracker in trackers.values():
                tracker.get_in_queue().put({'cmd': 'terminate'})
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

                in_queue = Queue()
                out_queue = Queue()

                trackers.update({tid: TrackerMP(True, False, True, in_queue, out_queue)})  # hog, fixed_window, multi-scale
                # if you use hog feature, there will be a short pause after you draw a first boundingbox,
                # that is due to the use of Numba.

                tracker = trackers.get(tid)
                tracker.start()
                tracker.get_in_queue().put({'cmd': 'init', 'roi': [ix, iy, w, h], 'image': frame})
                '''is_valid = tracker.init()
                if not is_valid:
                    del trackers[tid]
                    print('del')
                print(tid, 'is valid', frame.shape)'''

            initTracking = False
            onTracking = True
            # print('initTracking finished')

        elif onTracking:
            t0 = time()
            for (tid, tracker) in trackers.items():
                tracker.get_in_queue().put({'cmd': 'update', 'image': frame})

            for (tid, tracker) in trackers.items():
                # if tid in to_traced:
                # tracker.join()
                out_queue = tracker.get_out_queue()
                ret = out_queue.get()
                # print(ret)
                pid = ret[0]
                bbox = ret[1][0]
                pv = ret[1][1]
                # print(pv)
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

        if imshow_enable:
            cv2.imshow('tracking', frame)
            c = cv2.waitKey(1) & 0xFF
            if c == 27 or c == ord('q'):
                break

        cv2.imwrite('output/frame_%06d.jpg' % current_frame, frame)

        print()

    cap.release()
    cv2.destroyAllWindows()
