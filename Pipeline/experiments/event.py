import os
import sys
import csv
import time
import glob
import json
import shutil

from tqdm import tqdm

from src.utils import check_dir


## Day table
def get_times_detail(start_date_time, times):  # start_date_time : YYYYMMDDhhmmss, times : hhmmss
    DAY_TABLE = {1: '31', 2: 29 if (int(start_date_time[0:4]) % 4 == 0) else 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    date_time_info = [[int(start_date_time[0:4]), 10000], [int(start_date_time[4:6]), 12], [int(start_date_time[6:8]), DAY_TABLE[int(start_date_time[4:6])]], 
                      [int(start_date_time[8:10]), 24], [int(start_date_time[10:12]), 60], [int(start_date_time[12:14]), 60]]

    times_list = [int(times[:2]), int(times[2:4]), int(times[4:6])]
    for i, t in enumerate(reversed(times_list)):
        inx = -(i + 1)
        date_time_info[inx][0] += t
        while date_time_info[inx][0] >= date_time_info[inx][1]:
            next_value = int(date_time_info[inx][0] / date_time_info[inx][0])
            date_time_info[inx][0] = date_time_info[inx][0] % date_time_info[inx][1]
            inx -= 1
            date_time_info[inx][0] += next_value
    tl = [dt[0] for dt in date_time_info]
    detail_times = '%04d-%02d-%02d %02d:%02d:%02d' % (tl[0], tl[1], tl[2], tl[3], tl[4], tl[5])
    return detail_times


## prediction format
def convert_eval_format(events):
    """
    Use this func to convert predictions to annotations format
    """
    output = []
    # frames_list [[start_frame, end_frame, {frame (str): {'human_box': bbox}, ...}], ...] 
    for label, frames_list in events.items():
        if 'human' in label:
            human_id = int(label[label.index('human_') + 6:].split(' ')[0])
        else:
            human_id = -1
        for frame_list in frames_list:
            output.append([label, human_id, frame_list])
    return output

def produce_report_csv(output_path, save_path='', save_name='Report.csv'):
    # parse inference output event from video prediction
    event_json_paths = glob.glob(os.path.join(output_path, '*/*/*.json'))  # TODO : channels / datetime / .json

    output_info = {}
    pbar = tqdm(event_json_paths, total=len(event_json_paths))
    for event_json_path in pbar:
        channel = event_json_path.split('/')[-2]
        video_name = event_json_path.split('/')[-1].split('.')[0]
        date_times = video_name[:14]  # YYYYMMDDhhmmss
        prediction_dataset = json.load(open(event_json_path))
        for event_name, event_data in prediction_dataset['event_info'].items():
            if len(event_data['final']):
                events_pred = convert_eval_format(event_data['final'])
                for event_info in events_pred:
                    start_times = get_times_detail(date_times, event_info[2]['start_end_time'][0])  # hhmmss
                    end_times = get_times_detail(date_times, event_info[2]['start_end_time'][1])  # hhmmss

                    if channel not in output_info:
                        output_info[channel] = {}
                    

                    if date_times not in output_info[channel]:
                        output_info[channel][date_times] = [[start_times, end_times, event_name, event_info[0], event_info[1]]]
                    else:
                        output_info[channel][date_times].append([start_times, end_times, event_name, event_info[0], event_info[1]])
    # sort
    new_output_info = {}
    for ch, ch_info in output_info.items():
        new_output_info[ch] = {}
        ch_info_tuples = sorted(ch_info.items(), key=lambda x: x[0])
        for ch_info_tuple in ch_info_tuples:  # date_times, event_info
            new_output_info[ch][ch_info_tuple[0]] = sorted(ch_info_tuple[1], key=lambda x: [x[0], x[1]])
    print('------------------------------------------------------------------------------')


    # save csv file to Report 
    if save_path and save_name.split('.')[-1] == 'csv':
        check_dir(save_path)
        save_path_name = os.path.join(save_path, save_name)

        # pd.DataFrame(output).to_csv(os.path.join(self.save_path, 'output.csv'))
        with open(save_path_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write raw information
            for channel, ch_info in new_output_info.items():
                writer.writerow([channel])
                for date_times, event_infos in ch_info.items():
                    date_times = '%s-%s-%s %s:%s:%s' % (date_times[0:4], date_times[4:6], date_times[6:8], date_times[8:10], date_times[10:12], date_times[12:14])
                    writer.writerow(['', date_times, 'event_start_time', 'event_end_time', 'event', 'detail'])
                    for event_info in event_infos:
                        writer.writerow(['', '', event_info[0], event_info[1]])
                    writer.writerow([''])
                writer.writerow([''])
                writer.writerow([''])
        print('save output in > %s' % save_path_name)

    return output_info
