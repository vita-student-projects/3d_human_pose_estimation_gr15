import logging
import subprocess

logging.info("Evaluate EfficientNet")
subprocess.call("python -m romp.test --configs_yml=configs/eval_3dpw_test_effnet.yml", shell=True)
subprocess.call("python -m romp.test --configs_yml=configs/eval_cmu_panoptic_effnet.yml", shell=True)
