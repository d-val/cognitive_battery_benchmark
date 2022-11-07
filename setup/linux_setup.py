from tqdm import tqdm
import sys

import numpy as np
import os
import ai2thor.server
def start_xserver() -> None:
    """Provide the ability to render AI2-THOR using Google Colab."""

    # Thanks to the [Unity ML Agents team](https://github.com/Unity-Technologies/ml-agents)
    # for most of this setup! :)

    with tqdm(total=100) as pbar:

        with open("frame-buffer", "w") as writefile:
            writefile.write(
                """#taken from https://gist.github.com/jterrace/2911875
        XVFB=/usr/bin/Xvfb
        XVFBARGS=":1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset"
        PIDFILE=./frame-buffer.pid
        case "$1" in
        start)
            /sbin/start-stop-daemon --start --quiet --pidfile $PIDFILE --make-pidfile --background --exec $XVFB -- $XVFBARGS
            ;;
        stop)
            /sbin/start-stop-daemon --stop --quiet --pidfile $PIDFILE
            rm $PIDFILE
            ;;
        restart)
            $0 stop
            $0 start
            ;;
        *)
                exit 1
        esac
        exit 0
            """
            )

        pbar.update(5)
        os.system("apt-get install daemon >/dev/null 2>&1")

        pbar.update(5)
        os.system("apt-get install wget >/dev/null 2>&1")

        pbar.update(10)
        os.system(
            "wget http://ai2thor.allenai.org/ai2thor-colab/libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb >/dev/null 2>&1"
        )

        pbar.update(10)
        os.system(
            "wget --output-document xvfb.deb http://ai2thor.allenai.org/ai2thor-colab/xvfb_1.18.4-0ubuntu0.12_amd64.deb >/dev/null 2>&1"
        )

        pbar.update(10)
        os.system("dpkg -i libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb >/dev/null 2>&1")

        pbar.update(10)
        os.system("dpkg -i xvfb.deb >/dev/null 2>&1")

        pbar.update(20)
        os.system("rm libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb")

        pbar.update(10)
        os.system("rm xvfb.deb")

        pbar.update(10)
        os.system("bash frame-buffer start")

        os.environ["DISPLAY"] = ":1"
        pbar.update(10)

if __name__ == "__main__":
    start_xserver()
