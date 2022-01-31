import argparse
from ast import arg
from itertools import count
import os
import cv2

exp = "gravitybias"
app_path = "utils/GravityBias.app"
bin_path = app_path + '/Contents/MacOS/"Gravity Bias"'

def run_exp(exp, bin_path, show = True, record = True, outdir="output"):
    """
    Runs the Gravity Bias build.
    bin_path: (string) the path to the Unity binary build.
    show: (bool) whether to show the experiment.
    record: (bool) whether to record the experiment.
    outdir: (string) path to save recorded frames
    """
    args = ""
    if not show: args += " -batchmode "
    if record: args += "-record "
    if (record and len(outdir)): args += '--outdir "%s"' % outdir

    # Ensures the experiment app has execution permessions
    os.system("chmod +x ./%s" %bin_path)
    os.system("xattr -r -d com.apple.quarantine %s" %app_path)

    # Runs the experiment binary
    os.system('./%s %s' %(bin_path, args))

def save_video(frames_dir, video_path = "experiment_video.mp4"):
    """
    Turns the frames in frames_dir into a video and saves it at the video_path.
    frames_dir: (string) the path of the experiment frames.
    video_path: (string) the path at which the video is to be saved.
    """

    # Reading the frames from frames_dir
    images = [img for img in os.listdir(frames_dir) if img.endswith("png")]
    images.sort(key=lambda x:int(x[:-4].split("_")[-1]))
  
    frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, layers = frame.shape  

    # Creating the video at video_path
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(frames_dir, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()

if __name__ == "__main__":

    # Intilizating Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", help="Show experiment", action="store_true")
    parser.add_argument("-r", "--record", help="Record experiment", action="store_true")
    parser.add_argument("-o", "--outdir", help="Captures output directory", default="output", type=str)
    args = parser.parse_args()
    
    # Make sure that the Gravity Bias binary exists
    if not os.path.isdir(app_path):
        print("The Gravity Bias build is not in utils/ Please download it and re-run this script.")
    else:
        # Finding output path and creating appropriate directories
        human_readable_path = "./"
        if args.record:
            if not os.path.isdir(args.outdir):
                os.mkdir(args.outdir)
            expt_dir = args.outdir + "/Gravity Bias"
            if not os.path.isdir(expt_dir):
                os.mkdir(expt_dir)
            counter = 0
            for dir in os.listdir(expt_dir):
                if dir.isdigit():
                    counter = max(counter, int(dir) + 1)
            outpath = args.outdir + "/" + "Gravity Bias/" + str(counter)
            human_readable_path = outpath + "/human_readable"
            machine_readable_path = outpath + "/machine_readable"
            os.mkdir(outpath)
            os.mkdir(human_readable_path)
            os.mkdir(machine_readable_path)

        # Running experiment
        run_exp(exp, bin_path, args.show, args.record, "../%s" %human_readable_path)
    
    # Saving video from frames
    save_video(human_readable_path, video_path=os.path.join(outpath, "experiment_video.mp4"))