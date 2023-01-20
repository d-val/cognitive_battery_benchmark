from glob import glob
import yaml, json, sys

data_list = []
input_dict = {'left': '1', 'right': '2','center': '3'}
for folder in glob(f"{sys.argv[1]}/*/"):
    with open(f"{folder}/machine_readable/experiment_stats.yaml", 'r') as f:
        data = yaml.load(f)
    data_list.append({'stimulus': [f'{folder}experiment_video.mp4'], 'correct_response': input_dict[data[sys.argv[2]]] })

with open(f"processed_files.json", 'w') as f:
    data = json.dumps(data_list)
#     json.dump(data, f)
    f.write(data)
print(data_list)

