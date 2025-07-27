import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m%d%Y%H%M%S')}"  # ex o/p format : 07252025181830
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)    # getcwd() stands for "get current working directory".
# creating directory contains timestamp
os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level =logging.INFO,
)

# if __name__=='__main__':
#     logging.info("Logging has starting")
# Note if you want to use this file to another file then use give code
# import logger as logging