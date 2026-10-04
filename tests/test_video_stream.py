import os
import time
import pytest
import numpy as np
import cv2

from utils.video_stream import ThreadedVideoStream


@pytest.fixture
def temp_video(tmp_path):
    """Gecici 10 karelik test video dosyasi olusturur."""
    video_path = str(tmp_path / "test_stream.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (120, 100))

    for i in range(10):
        frame = np.full((100, 120, 3), fill_value=i * 20, dtype=np.uint8)
        out.write(frame)

    out.release()
    return video_path


class TestThreadedVideoStream:
    def test_invalid_source(self):
        """Var olmayan video kaynagi acilamamali ve hata vermeli."""
        stream = ThreadedVideoStream("non_existent_video_file_xyz.mp4")
        assert not stream.opened
        assert not stream.is_opened()

        with pytest.raises(RuntimeError):
            stream.start()

        ret, frame = stream.read()
        assert not ret
        assert frame is None

        stream.stop()

    def test_read_video_file_complete(self, temp_video):
        """Tum video karelerinin kayipsiz ve sirali okunmasi."""
        with ThreadedVideoStream(temp_video, queue_size=16, is_live=False) as stream:
            assert stream.opened
            assert stream.is_opened()
            assert stream.width == 120
            assert stream.height == 100

            frames = []
            while True:
                ret, frame = stream.read(timeout=1.0)
                if not ret:
                    break
                frames.append(frame)

            assert len(frames) == 10
            for idx, f in enumerate(frames):
                assert f.shape == (100, 120, 3)
                # Piksel degerlerinin sirali oldugunu dogrula (kayipli video codec payi ile)
                assert np.mean(f) == pytest.approx(idx * 20, abs=5.0)

            # Video bitince False donmeli
            ret_end, frame_end = stream.read(timeout=0.1)
            assert not ret_end
            assert frame_end is None

    def test_live_mode_behavior(self, temp_video):
        """Canli kamera modunda arka plan okumasi ve guvenli durdurma."""
        stream = ThreadedVideoStream(temp_video, is_live=True)
        assert stream.is_live
        stream.start()

        # Bir kare oku
        ret, frame = stream.read(timeout=1.0)
        assert ret
        assert frame is not None

        # Guvenli durdurma
        stream.stop()
        assert not stream.opened
        assert not stream.is_opened()

    def test_stop_idempotency(self, temp_video):
        """stop() metodunun coklu cagrilmasi guvenli olmalidir."""
        stream = ThreadedVideoStream(temp_video)
        stream.start()
        stream.stop()
        # Tekrar cagrildiginda hata vermemeli
        stream.stop()
        assert not stream.opened
