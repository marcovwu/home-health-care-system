import os
import cv2
import glob
import json
import copy
import numpy as np

from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from configs.configs import PipelineConfigs as Configs
from utils.metrics import bbox_ioa
from collections import deque


def make_ydt(year_date_time):
    if len(year_date_time) == 12:
        return '20' + year_date_time
    return year_date_time


def sec_to_times(sec):
    second = sec % 60
    minute = int(sec / 60) % 60
    hour = int(sec / 3600)
    return '%02d%02d%02d' % (hour, minute, second)

def fp_filter(length, filter_frame):
    if length < filter_frame:
        return True
    return False

def inter_point(x, y, xyxy, close):
    if x > xyxy[0] - close and x < xyxy[2] + close and y > xyxy[1] - close and y < xyxy[3] + close:
        return True

class EventGenerator:
    WRITE_SEC = 10
    OBJS = ['human']
    STATE = {'w': 'walking', 's': 'stop', 'm': 'miss', 'n': 'none'}
    MODE = {'w': 'work', 's': 'stop', 'o': 'off'}
    SEG_INFO = {'img': {Path(img_path).name: cv2.imread(img_path) for img_path in glob.glob(os.path.join(Configs.seg_path, '*.png'))}, 'info': json.load(open(os.path.join(Configs.seg_path, 'seg.json')))}

    def __init__(self, of, video_path, fps=25, maxframes=7550, save_dir=''):
        self.of = of
        self.fps = fps
        self.maxframes = maxframes
        self.video_path = video_path
        self.channel = video_path.split('/')[-2]
        self.seg_img = self.SEG_INFO['img'][self.channel + "_seg.png"]

        self.save_dir = save_dir
        self.video_name = Path(video_path).name
        self.start_time = make_ydt(self.video_name.split('.')[0])
        self.events = {
            'human_lie_event': {'final': {}, 'show_frame': 10, 'candidate_lie': False, 'start_frame': -1, 'success_second': 10, 'record': {}}}
        self.config = {'human': {'frames_info': {'least_frames': 20, 'thres': 0.8, 'maxframes': 50, 'last_frame_gap': 50, 'box_info': {}}}}

    def clean_past_objandevent(self, frame):
        """
        Clean tracklet if detection miss for a long time
        """
        # """
        # clean object id
        for obj, obj_info in self.config.items():
            del_candidate = []
            for _id, id_info in obj_info['frames_info']['box_info'].items():
                if frame - id_info[-1][5] >= self.config[obj]['frames_info']['last_frame_gap']:  # check last box's frame gap
                    del_candidate.append(_id)
            # delete
            for dc_id in del_candidate:
                self.config[obj]['frames_info']['box_info'].pop(dc_id)
                # """
        # clean event
        for event_name, event_info in self.events.items():
            if frame - event_info['show_frame'] in event_info['record']:
                event_info['record'].pop(frame - event_info['show_frame'])


    def object_state(self, frame, _id, bbox, cls='Lie', obj='human'):
        """
        :param _id: tracking id
        :param bbox: id's bounding box
        :return: object state
        """
        # working state
        if _id in self.config[obj]['frames_info']['box_info']:
            if frame - self.config[obj]['frames_info']['box_info'][_id][-1][5] >= self.config[obj]['frames_info']['last_frame_gap']:  # check last box's frame gap
                state = self.STATE["n"]
                self.config[obj]['frames_info']['box_info'][_id] = deque([(*bbox, state, frame)])
            elif len(self.config[obj]['frames_info']['box_info'][_id]) >= self.config[obj]['frames_info']['maxframes']:  # check max frames numbers
                self.config[obj]['frames_info']['box_info'][_id].popleft()

            # get min box to be iou denominator
            x1, y1, x2, y2 = self.config[obj]['frames_info']['box_info'][_id][0][:4]
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[0]) < (x2 - x1) * (y2 - y1):
                box2 = np.array(bbox)
                box1 = np.array(self.config[obj]['frames_info']['box_info'][_id][0][:4])
            else:  # ori mode
                box1 = np.array(bbox)
                box2 = np.array(self.config[obj]['frames_info']['box_info'][_id][0][:4])

            # check new state
            if len(self.config[obj]['frames_info']['box_info'][_id]) < self.config[obj]['frames_info']['least_frames']:
                state = self.STATE["n"]
            elif bbox_ioa(box1, box2) > self.config[obj]['frames_info']['thres']:
                state = self.STATE["s"]
            else:
                state = self.STATE["w"]
            self.config[obj]['frames_info']['box_info'][_id].append((*bbox, state, frame))
        else:
            state = self.STATE["n"]
            self.config[obj]['frames_info']['box_info'][_id] = deque([(*bbox, state, frame)])

        # check events to final json
        if obj == 'human':
            self.human_lie_event(frame, _id, bbox, cls)

        return state


    def human_lie_event(self, frame, _id, bbox, cls):
        """
        :return: detect whether a human lie in x seconds
        """
        if cls != 'Lie':
            self.events['human_lie_event']['candidate_lie'] = False
            return False
        if self.events['human_lie_event']['candidate_lie'] and \
            (frame - self.events['human_lie_event']['start_frame']) / self.fps > self.events['human_lie_event']['success_second']:
            x1, y1, x2, y2 = [float(_) for _ in bbox]
            event_str = 'Event : human_%s lie event' % str(_id)
            # final
            if event_str in self.events['human_lie_event']['final']:
                self.events['human_lie_event']['final'][event_str][frame] = {'human_box': [x1, y1, x2, y2]}
            else:
                self.events['human_lie_event']['final'][event_str] = {frame: {'human_box': [x1, y1, x2, y2]}}
            # check show event
            if frame in self.events['human_lie_event']['record']:
                self.events['human_lie_event']['record'][frame].append(event_str)
            else:
                self.events['human_lie_event']['record'][frame] = [event_str]
            return True
        elif not self.events['human_lie_event']['candidate_lie']:
            self.events['human_lie_event']['candidate_lie'] = True
            self.events['human_lie_event']['start_frame'] = frame

        return False


    def send_email(self,
                   email='test@gmail.com',
                   subject='Testing Mail from Dante',
                   content='This is a testing mail sent from Dante.'):
        comm = 'echo "' + content + '" | mail -s "' + subject + '" ' + email
        os.system(comm)

        print('send an email to {}'.format(email))


    def show_result(self, imc, frame):
        """
        Use this func to write in the video
        """
        # check result
        self.clean_past_objandevent(frame)

        # show human_lie_event result
        human_idx = 2
        human_lie_event = [string for str_list in self.events["human_lie_event"]["record"].values() for string in str_list]
        for human_idx, event_str in enumerate(list(set(human_lie_event))):
            cv2.putText(imc, event_str, (500, human_idx * 30 + 30), 0, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


    def write_events_result(self, gap=75):
        """
        Use this func to write events to json file
        """
        # process event > [start frame, end frame]
        for event_name, info in self.events.items():
            for event_str, frames_info in info["final"].items():
                sta_f = {}
                frame_list = []
                for inx, frame in enumerate(sorted(list(frames_info.keys()), key=lambda x:x)):
                    frame_time = sec_to_times(round(frame / self.fps))
                    if len(sta_f) == 0:
                        sta_f = {'start_end_frame': [frame, frame], 
                                 'start_end_time': [frame_time, frame_time],
                                 'frame_info': {frame: frames_info[frame]}}
                    elif frame - sta_f['start_end_frame'][1] < gap:
                        sta_f['start_end_frame'][1] = frame
                        sta_f['start_end_time'][1] = frame_time
                        # add box info to frame_list
                        sta_f['frame_info'][frame] = frames_info[frame]
                        # check last frame
                        if inx == len(frames_info) - 1:
                            frame_list.append(sta_f)
                    else:
                        frame_list.append(sta_f)
                        sta_f = {'start_end_frame': [frame, frame], 
                                 'start_end_time': [frame_time, frame_time],
                                 'frame_info': {frame: frames_info[frame]}}
                
                self.events[event_name]["final"][event_str] = frame_list
        # save video info to output json
        output = {'video_info': {'fps': self.fps, 'maxframes': self.maxframes, 'video_path': self.video_path, 'channel': self.channel}}
        output['event_info'] = copy.deepcopy(self.events)
        with open(str(self.save_dir / (self.start_time + '.json')), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


    def save_event_video(self, send_fromend=False):
        """
        Use this func to cut final events video from output video
        """
        # TODO : only agv stop event
        event_names = ['human_lie_event']

        # Save results (image with detections)
        whole_event_str = []
        whole_video_name = str(self.save_dir / self.video_name)
        whole_video_name = str(Path(whole_video_name).with_suffix(self.of))
        for event_name in event_names:
            for event_str, frame_list in self.events[event_name]["final"].items():
                for frame_infos in frame_list:
                    startsec = round(max(frame_infos['start_end_frame'][0] / self.fps - 3, 0))
                    endsec = round(min(max(frame_infos['start_end_frame'][1] / self.fps + 3, startsec + self.WRITE_SEC), self.maxframes / self.fps))
                    save_path = str(self.save_dir / (self.start_time + '_' + sec_to_times(startsec) + '-' + sec_to_times(endsec) + event_str + self.of))  # date_time + startsec + endsec + output_video_format
                    print(save_path)
                    whole_event_str.append(self.start_time + '_' + sec_to_times(startsec) + '-' + sec_to_times(endsec) + '\n' + event_str)
                    # cut video
                    ffmpeg_extract_subclip(whole_video_name, startsec, endsec, targetname=save_path)  # start second to end second
        # send email
        if send_fromend: self.send_email(email=Configs.email, subject='[Emgerency Event]', content='\n\n'.join(whole_event_str))