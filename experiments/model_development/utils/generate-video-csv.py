import cv2
from pathlib import Path

print('started generating video csv')

def get_duration(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return num_frames / fps

with open('video_csv.csv', 'w') as f:
    f.write('video_path,start_time,end_time')
    for video_path in Path('..').glob('data/*/*.mp4'):
        print('found video_path', video_path)
        start_time = 0
        end_time = get_duration(str(video_path))

        f.write('\n' + str(video_path) + ',' + str(start_time) + ',' + str(end_time))

print('done generating video csv')
