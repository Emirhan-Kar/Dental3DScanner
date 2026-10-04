import os
import time
import queue
import threading
import cv2
import numpy as np
from typing import Tuple, Optional, Union

from utils.logger import get_logger
from utils.validator import get_safe_path

logger = get_logger(__name__)


class ThreadedVideoStream:
    """
    Coklu Is Parcacikli (Asenkron) Video Yakalama Sinifi.

    Gercek zamanli intraoral dental tarama sirasinda agir 3D hesaplamalar (ICP vb.)
    yapilirken kamera donanim tamponunun tasmasini ve kare atlamalarini onlemek icin
    kamera okuma islemini ayri bir arka plan thread'inde yurutur.

    Ozellikler:
    - Canli kamera modu: Gecikmeyi onlemek icin en guncel kareyi tutar, eski kareleri atar.
    - Video dosya modu: Rekonstruksiyon butunlugu icin hicbir kareyi kaybetmeden kuyruklar.
    - Context manager (with) destegi.
    - Windows Unicode/Turkce dosya yollari ile tam uyumluluk (get_safe_path).
    """

    def __init__(
        self,
        src: Union[int, str] = 0,
        queue_size: int = 64,
        is_live: Optional[bool] = None,
        name: str = "ThreadedVideoStream"
    ):
        """
        Args:
            src (int veya str): Kamera indeksi (orn: 0) veya video dosya yolu.
            queue_size (int): Arka plan kare kuyrugu azami boyutu.
            is_live (bool, optional): Canli kamera modu. Belirtilmezse kaynaga gore otomatik belirlenir.
            name (str): Thread adi.
        """
        self.src = src
        self.name = name

        # Kaynak tipini ve canli akis modunu tespit et
        self.is_cam = False
        if isinstance(src, int):
            self.is_cam = True
            safe_src = src
        elif isinstance(src, str) and src.isdigit():
            self.is_cam = True
            safe_src = int(src)
        else:
            self.is_cam = False
            safe_src = get_safe_path(str(src))

        self.is_live = is_live if is_live is not None else self.is_cam
        # Canli kamera durumunda dusuk gecikme icin kucuk kuyruk
        actual_queue_size = 2 if self.is_live else max(1, queue_size)

        self.cap = cv2.VideoCapture(safe_src)
        if not self.cap.isOpened():
            logger.error(f"Video kaynagi acilamadi: {src}")
            self.opened = False
        else:
            self.opened = True

        # Video metadata
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self.queue = queue.Queue(maxsize=actual_queue_size)
        self._stop_event = threading.Event()
        self._thread = None
        self.frames_read = 0
        self.frames_dropped = 0

    def start(self) -> "ThreadedVideoStream":
        """Arka plan okuma thread'ini baslatir."""
        if not self.opened:
            raise RuntimeError(f"Video kaynagi acik degil: {self.src}")

        if self._thread is not None and self._thread.is_alive():
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name=self.name, daemon=True)
        self._thread.start()
        logger.debug(f"{self.name} baslatildi (Kaynak: {self.src}, Canli: {self.is_live})")
        return self

    def _worker(self):
        """Arka planda calisan kare okuma dongusu."""
        while not self._stop_event.is_set():
            if not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Video bitti veya kare alinamadi
                break

            self.frames_read += 1

            if self.is_live:
                # Canli kamera: Kuyruk doluysa en eski kareyi cikarip en yeniyi koy (sifir gecikme)
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass
                try:
                    self.queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass
            else:
                # Video dosyasi: Hicbir kareyi kaybetmeden kuyruga ekle
                while not self._stop_event.is_set():
                    try:
                        self.queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        self._stop_event.set()

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Kuyruktan bir sonraki kareyi ceker.

        Args:
            timeout (float): Bekleme zaman asimi (saniye).

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (Basarili_mi, Kare_Goruntusu).
        """
        if not self.opened:
            return False, None

        try:
            frame = self.queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            if self._stop_event.is_set():
                return False, None
            return False, None

    def is_opened(self) -> bool:
        """Stream'in acik ve calisabilir durumda olup olmadigini dondurur."""
        return self.opened and (not self._stop_event.is_set() or not self.queue.empty())

    def stop(self):
        """Thread'i durdurur ve video yakalayiciyi serbest birakir."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.opened = False
        logger.debug(f"{self.name} durduruldu (Okunan: {self.frames_read}, Atlanan: {self.frames_dropped})")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
