from mmaction.apis import init_recognizer, inference_recognizer
config_file = '../configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
# build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device='cpu')
# test a single video and show the result:
video = 'demo.mp4'
label = 'label_map_k400.txt'
results = inference_recognizer(model, video, label)
# show the results
for result in results:
    print(f'{result[0]}: ', result[1])
