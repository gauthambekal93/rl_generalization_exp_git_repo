41# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:47:36 2024

@author: gauthambekal93
"""



'''
import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games")

#os.system("python temp_file.py")

import subprocess
subprocess.run(["python", "temp_file.py"])
'''
'''
import subprocess
import time

def run_script():
    while True:
        try:
            # Run your main script
            result = subprocess.run(["python", "temp_file.py"], check=True)
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Script crashed with error: {e}")
            print("Restarting script in 20 seconds...")
            #time.sleep(20)
            break

if __name__ == "__main__":
    run_script()
'''


import subprocess
import sys
import time
import os

def run_script():
    # Get the full path to the Python interpreter
    python_executable = r"C:/Users/gauthambekal93/Research/rl_generalization_exps/myenv_generalization_tasks/Scripts/python.exe"
    script_path = os.path.abspath(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Code_v3.py")
    
    while True:
        try:
            # Run your main script and capture output and errors
           result = subprocess.Popen(
                [python_executable,"-u", script_path],
                #check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line-buffered
            )
           print("Output:", result.stdout)
           #break  # Exit the loop if the script runs successfully
            
            
            # Print the output and errors in real-time
           with result.stdout, result.stderr:
               for stdout_line in iter(result.stdout.readline, ""):
                   print(stdout_line, end="")  # end="" prevents adding extra newline
               for stderr_line in iter(result.stderr.readline, ""):
                   print(stderr_line, end="")  # end="" prevents adding extra newline

           #result.stdout.close()
           #result.stderr.close()
           result.wait()

           if result.returncode == 0:
               print("Script completed successfully")
               break
           else:
               raise subprocess.CalledProcessError(result.returncode, [python_executable, script_path])



        except subprocess.CalledProcessError as e:
            # Print the error details
            print(f"Script crashed with error: {e.stderr}")
            print("Restarting script in 20 seconds...")
            
            # Log the error details to a file
            with open("error_log.txt", "a") as log_file:
                log_file.write(f"Script crashed with error:\n{e.stderr}\n")
                log_file.write(f"Output before crash:\n{e.stdout}\n")
            
            # Wait before restarting the script
            time.sleep(20)

if __name__ == "__main__":
    run_script()
