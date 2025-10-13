import logging
from datetime import datetime
import pytz
from pathlib import Path
from logging.handlers import RotatingFileHandler

path_root_dir = Path(__file__).parent.parent
if not (path_root_dir / "logs").exists():
    (path_root_dir / "logs").mkdir()
path_log = path_root_dir / "logs" / "app.log"
#log_level = logging.DEBUG  # 기본 로그 레벨 설정
log_level = logging.INFO  # 기본 로그 레벨 설정


# 로거 설정
logger = logging.getLogger("MyLogger")
logger.setLevel(log_level)

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

# 파일 핸들러 추가
file_handler = RotatingFileHandler(
    path_log,
    maxBytes=100 * 1024 * 1024,  # 최대 파일 크기 (5MB)
    backupCount=3,  # 보관할 최대 파일 개수
)
file_handler.setLevel(log_level)


class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone("Asia/Seoul"))
        return dt.strftime(datefmt) if datefmt else dt.isoformat()


# 포맷 설정
formatter = KSTFormatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 로거에 핸들러 추가
logger.addHandler(console_handler)
logger.addHandler(file_handler)