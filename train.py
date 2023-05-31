import logging
import subprocess

logging.info("Start Training")
subprocess.call("./scripts/V1_train_effnet.sh", shell=True)
