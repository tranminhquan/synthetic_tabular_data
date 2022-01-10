import pandas as pd
import os
import sys
import subprocess
import threading
import time

TEMPLATE = "kaggle datasets download -d {}"

class Command(object):
    def __init__(self):
        self.process = None

    def run(self, cmd, timeout):
        def target():
            self.process = subprocess.Popen(cmd, shell=True)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        time.sleep(10)

        if thread.is_alive():
            print('Terminating process')
            self.process.terminate()
            self.process = None
            thread.join()
        print(self.process.returncode)

if __name__=="__main__":
    assert (len(sys.argv) == 2) and (".csv" in sys.argv[1]),\
        "use \"python download.py <path_to_csv>\" to run download...."
    command = Command()
    df = pd.read_csv(sys.argv[1])
    
    for line in df.url.to_list():
        print(line)
        command.run(TEMPLATE.format(line[1:]), timeout=10)
        
