from __future__ import print_function

import cv2
import sys
from time import time
import os

from multiprocessing import Queue
from tracker_mp import TrackerMP
from config import *
from util import *


def parse_label(label_line, data_format=''):
    if data_format == 'KITTI':
        val = label_line.split(' ')
        d = {'frame': int(val[0]), 'id': int(val[1]), 'object_class': val[2].strip('"'),
             'x1': int(float(val[6])), 'y1': int(float(val[7])), 'x2': int(float(val[8])), 'y2': int(float(val[9]))}
    elif data_format == 'VTB':
        val = label_line.split('\t')
        d = {'frame': int(val[0]) - 1, 'id': int(val[1]), 'object_class': val[2].strip('"'),
             'x1': int(float(val[3])), 'y1': int(float(val[4])),
             'x2': int(float(val[3]) + float(val[5])), 'y2': int(float(val[4]) + float(val[6]))}
    else:
        val = label_line.split('\t')
        d = {'frame': int(val[0].lstrip('0')) - 1, 'id': int(val[1]), 'object_class': val[2].strip('"'),
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

    if input_mode == 'video':
        cap = cv2.VideoCapture(input_v_path)
        assert cap.isOpened()
        frames_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        files_in_dir = sorted([f for f in os.listdir(input_v_path) if os.path.isfile(os.path.join(input_v_path, f))])
        frames_count = len(files_in_dir)

    frames = list()
    for _ in range(frames_count):
        frames.append(list())
    with open(labels_file) as labels:
        for line in labels.readlines():
            info = parse_label(line, 'KITTI')
            if info.get('id') != -1:  # pass unknown objects
                fi = info.get('frame')
                frames[fi].append(info)

    all_trackers = dict()
    tracker_valid = dict()
    duration = 0.01
    duration_smooth = 0.01
    sum_pv = 0.0
    no_result_count = 0

    # ============  main tracking loop  ============ #
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

        # select mode
        initTracking = False
        if current_frame % detection_period == 0:
            initTracking = True

        if initTracking:
            # invalidate old trackers
            for tid, tracker in all_trackers.items():
                tracker_valid.update({tid: False})

            all_targets = sorted(frames[current_frame], key=lambda d: d.get('id'))
            print(' ---------------- # trackers = %d' % len(all_targets), end='')
            for target in all_targets:
                # print(target)
                ix = target.get('x1')
                iy = target.get('y1')
                w = target.get('x2') - ix
                h = target.get('y2') - iy
                tid = target.get('id')
                object_class = target.get('object_class')

                tracker = all_trackers.get(tid)
                if tracker is None:
                    tracker = TrackerMP(hog=True, fixed_window=False, multi_scale=True,
                                        input_queue=Queue(), output_queue=Queue())
                    tracker.start()
                tracker.get_in_queue().put({'cmd': 'init',
                                            'object_class': object_class,
                                            'roi': [ix, iy, w, h],
                                            'image': frame})
                all_trackers.update({tid: tracker})  # add to trackers' dict
                tracker_valid.update({tid: True})

            initTracking = False
            onTracking = True

        elif onTracking:
            t0 = time()
            for (tid, tracker) in all_trackers.items():
                tracker.get_in_queue().put({'cmd': 'update',
                                            'image': frame})
            # trackers  will calculate in their sub-processes
            for tid, tracker in all_trackers.items():
                if not tracker_valid.get(tid):
                    continue
                ret = tracker.get_out_queue().get()
                if ret is None:
                    # something wrong with this tracker, pass it
                    print('ret == None, tid = %d' % tid)
                    no_result_count += 1
                    continue
                roi = ret.get('roi')
                pv = ret.get('pv')
                sum_pv += pv
                object_class = ret.get('object_class')
                bbox = map(int, roi)

                if imshow_enable or imwrite_enable:
                    if pv > 0.25:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 255), 1)
                        cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 1)
                        cv2.putText(frame, '%d-%s' % (tid, object_class), (bbox[0], bbox[1] + bbox[3] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, '%.2f' % pv, (bbox[0], bbox[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 0, 0), 1)
            t1 = time()
            duration = t1 - t0
            duration_smooth = 0.8 * duration_smooth + 0.2 * (t1 - t0)
            fps = 1 / duration_smooth
            print(' fsp = %4f' % fps, end='')
            if imshow_enable or imwrite_enable:
                cv2.putText(frame, 'FPS: ' + str(fps)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

        if imshow_enable:
            cv2.imshow('tracking', frame)
            c = cv2.waitKey(1) & 0xFF
            if c == 27 or c == ord('q'):
                break

        if imwrite_enable:
            cv2.imwrite('output/frame_%06d.jpg' % current_frame, frame)

        print(' sum_pv = %f' % sum_pv)

    # terminate all trackers after all frames are processed
    for tid, tracker in all_trackers.items():
        tracker.get_in_queue().put({'cmd': 'terminate'})
        if tracker.is_alive():
            tracker.terminate()

    if input_mode == 'video':
        cap.release()
    if imshow_enable:
        cv2.destroyAllWindows()

    print('no_result_count = %d' % no_result_count)
