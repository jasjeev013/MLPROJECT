import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path = os.path.join(os.getcwd(), 'logs',LOG_FILE)
os.makedirs(logs_path , exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] [%(message)s]",
    level=logging.INFO,
)

if __name__=="__main__":
    logging.info("This is a test log message")