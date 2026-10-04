import os
import sys
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "dental_scanner.log")
DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class SafeStreamHandler(logging.StreamHandler):
    """
    Windows cp1254 veya sinirli karakter kumesine sahip terminallerde
    UnicodeEncodeError olusmasini engelleyen guvenli StreamHandler.
    """
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                # Desteklenmeyen karakterleri guvenle degistir
                safe_msg = msg.encode(self.stream.encoding or "ascii", errors="replace").decode(self.stream.encoding or "ascii")
                self.stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)


def get_log_level_from_env(default_level=logging.INFO):
    """Cevresel degiskenden (DENTAL_LOG_LEVEL) log seviyesini okur."""
    env_level = os.getenv("DENTAL_LOG_LEVEL", "").upper().strip()
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return levels.get(env_level, default_level)


def setup_logger(
    name="DentalScanner",
    log_file=DEFAULT_LOG_FILE,
    level=None,
    console=True,
    rotate=True,
    max_bytes=10 * 1024 * 1024,  # 10 MB
    backup_count=5
):
    """
    Proje genelinde kullanilacak merkezi logger'i yapilandirir.

    Args:
        name (str): Logger adi veya hiyerarsik modul adi.
        log_file (str): Log dosyasinin kaydedilecegi yol.
        level (int): logging.DEBUG, INFO vb. None ise env degiskeni veya INFO kullanilir.
        console (bool): Konsola log basilip basilmayacagi.
        rotate (bool): Dosya boyutuna gore rotasyon yapilip yapilmayacagi.
        max_bytes (int): Log dosyasinin ulasabilecegi maksimum bayt boyutu.
        backup_count (int): Saklanacak gecmis log dosyasi adedi.

    Returns:
        logging.Logger: Yapilandirilmis logger ornegi.
    """
    if level is None:
        level = get_log_level_from_env(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Zaten handler tanimlanmissa tekrar ekleme (duplicate log onleme)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Konsol Handler (Guvenli cikis)
    if console:
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 2. Dosya Handler (Rotasyonlu)
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            if rotate:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8"
                )
            else:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")

            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Log dosyasi olusturulamadi ({log_file}): {e}")

    # Root logger'a yayilmasini kontrol et
    logger.propagate = False

    return logger


def get_logger(module_name=None):
    """
    Moduller icin hazir logger nesnesi dondurur.
    Eger merkezi logger kurulmadiysa otomatik yapilandirir.

    Kullanim:
        from utils.logger import get_logger
        logger = get_logger(__name__)
    """
    base_logger = logging.getLogger("DentalScanner")
    if not base_logger.handlers:
        setup_logger("DentalScanner")

    if not module_name:
        return base_logger

    child_logger = logging.getLogger(f"DentalScanner.{module_name}")
    child_logger.setLevel(base_logger.level)
    return child_logger

