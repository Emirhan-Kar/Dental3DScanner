import os
import sys
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_DIR = "logs"
VARSAYILAN_LOG_DIZINI = DEFAULT_LOG_DIR

DEFAULT_LOG_FILE = os.path.join(VARSAYILAN_LOG_DIZINI, "dental_scanner.log")
VARSAYILAN_LOG_DOSYASI = DEFAULT_LOG_FILE

DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
VARSAYILAN_LOG_FORMATI = DEFAULT_LOG_FORMAT

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
TARIH_FORMATI = DATE_FORMAT


class SafeStreamHandler(logging.StreamHandler):
    """
    Windows cp1254 veya sinirli karakter kumesine sahip terminallerde
    UnicodeEncodeError olusmasini engelleyen guvenli StreamHandler.
    """
    def emit(self, kayit):
        try:
            super().emit(kayit)
        except UnicodeEncodeError:
            try:
                metin = self.format(kayit)
                kodlama = self.stream.encoding or "ascii"
                guvenli_metin = metin.encode(kodlama, errors="replace").decode(kodlama)
                self.stream.write(guvenli_metin + self.terminator)
                self.flush()
            except Exception:
                self.handleError(kayit)


def get_log_level_from_env(varsayilan_seviye=logging.INFO):
    """Cevresel degiskenden (DENTAL_LOG_LEVEL) log seviyesini okur."""
    ortam_seviyesi = os.getenv("DENTAL_LOG_LEVEL", "").upper().strip()
    seviyeler = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return seviyeler.get(ortam_seviyesi, varsayilan_seviye)


def setup_logger(
    isim="DentalScanner",
    log_dosyasi=VARSAYILAN_LOG_DOSYASI,
    seviye=None,
    konsol=True,
    rotasyon=True,
    maks_bayt=10 * 1024 * 1024,  # 10 MB
    yedek_sayisi=5,
    **ek_parametreler
):
    """
    Proje genelinde kullanilacak merkezi kayitciyi (logger) yapilandirir.

    Args:
        isim (str): Kayitci adi veya modul hiyerarsisi.
        log_dosyasi (str): Log dosyasinin kaydedilecegi dosya yolu.
        seviye (int): logging.DEBUG, INFO vb.
        konsol (bool): Konsola log basilip basilmayacagi.
        rotasyon (bool): Dosya boyutuna gore rotasyon yapilip yapilmayacagi.
        maks_bayt (int): Bir log dosyasinin ulasabilecegi azami bayt boyutu.
        yedek_sayisi (int): Saklanacak gecmis log dosyasi sayisi.

    Returns:
        logging.Logger: Yapilandirilmis logger nesnesi.
    """
    isim = ek_parametreler.get("name", isim)
    log_dosyasi = ek_parametreler.get("log_file", log_dosyasi)
    seviye = ek_parametreler.get("level", seviye)
    konsol = ek_parametreler.get("console", konsol)
    rotasyon = ek_parametreler.get("rotate", rotasyon)
    maks_bayt = ek_parametreler.get("max_bytes", maks_bayt)
    yedek_sayisi = ek_parametreler.get("backup_count", yedek_sayisi)

    if seviye is None:
        seviye = get_log_level_from_env(logging.INFO)

    kayitci = logging.getLogger(isim)
    kayitci.setLevel(seviye)

    # Zaten handler tanimlanmissa tekrar ekleme (cift log yazimini onler)
    if kayitci.handlers:
        return kayitci

    bicimlendirici = logging.Formatter(fmt=VARSAYILAN_LOG_FORMATI, datefmt=TARIH_FORMATI)

    # 1. Konsol Handler (Guvenli cikis)
    if konsol:
        konsol_isleyici = SafeStreamHandler(sys.stdout)
        konsol_isleyici.setLevel(seviye)
        konsol_isleyici.setFormatter(bicimlendirici)
        kayitci.addHandler(konsol_isleyici)

    # 2. Dosya Handler (Rotasyonlu)
    if log_dosyasi:
        try:
            dizin_yolu = os.path.dirname(log_dosyasi)
            if dizin_yolu:
                os.makedirs(dizin_yolu, exist_ok=True)

            if rotasyon:
                dosya_isleyici = RotatingFileHandler(
                    log_dosyasi,
                    maxBytes=maks_bayt,
                    backupCount=yedek_sayisi,
                    encoding="utf-8"
                )
            else:
                dosya_isleyici = logging.FileHandler(log_dosyasi, encoding="utf-8")

            dosya_isleyici.setLevel(seviye)
            dosya_isleyici.setFormatter(bicimlendirici)
            kayitci.addHandler(dosya_isleyici)
        except Exception as hata:
            kayitci.warning(f"Log dosyasi olusturulamadi ({log_dosyasi}): {hata}")

    # Loglarin kok (root) kayitciya yayilmasini engelle
    kayitci.propagate = False

    return kayitci


def get_logger(modul_adi=None, **ek_parametreler):
    """
    Moduller icin hazir kayitci nesnesi dondurur.
    Eger merkezi kayitci kurulmadiysa otomatik yapilandirir.
    """
    modul_adi = ek_parametreler.get("module_name", modul_adi)

    kok_kayitci = logging.getLogger("DentalScanner")
    if not kok_kayitci.handlers:
        setup_logger("DentalScanner")

    if not modul_adi:
        return kok_kayitci

    alt_kayitci = logging.getLogger(f"DentalScanner.{modul_adi}")
    alt_kayitci.setLevel(kok_kayitci.level)
    return alt_kayitci
