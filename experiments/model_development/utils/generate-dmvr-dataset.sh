# run before train.py!
python3 generate-video-csv.py
#
brew install ffmpeg
pip3 install ffmpeg-python
#
mkdir dmvr_dataset
python3 generate_from_file.py --csv_path=data_video.csv --output_path=dmvr_dataset
