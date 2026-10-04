import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Proje kok dizinini sys.path'e ekle
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ai_model.dataset import DentalMeshDataset
from ai_model.network import DentalPointNetSeg
from ai_model.metrics import evaluate_segmentation_batch
from utils.logger import get_logger
from utils.config import load_config, get_base_parser

logger = get_logger(__name__)


def train(
    config=None,
    num_classes=None,
    num_points=None,
    batch_size=None,
    epochs=None,
    learning_rate=None,
    checkpoint_path=None,
    data_parts=None,
    lr_step_size=None,
    lr_gamma=None,
    resume_path=None
):
    """
    3D Diş Nokta Bulutu Segmentasyon Ağı Eğitim Fonksiyonu.
    Parametreler verilmezse `configs/config.yaml` dosyasından otomatik yüklenir.

    Args:
        config (ConfigNode veya dict, optional): Konfigurasyon nesnesi.
        num_classes (int, optional): Sınıf adedi.
        num_points (int, optional): Her örnekten alınacak nokta sayısı.
        batch_size (int, optional): Mini-batch boyutu.
        epochs (int, optional): Toplam epoch sayısı.
        learning_rate (float, optional): Öğrenme katsayısı.
        checkpoint_path (str, optional): Model ağırlıklarının kaydedileceği yol.
        data_parts (list, optional): Eğitime dahil edilecek data_part klasörleri listesi.
        lr_step_size (int, optional): StepLR adim araligi.
        lr_gamma (float, optional): StepLR gamma katsayisi.
    """
    # Konfigürasyon nesnesi yoksa varsayılanı yükle
    if config is None:
        config = load_config()

    ai_cfg = config.ai_model

    # Parametreleri config ile harmanla (Fonksiyon argümanı varsa önceliklidir)
    num_classes = num_classes if num_classes is not None else ai_cfg.num_classes
    num_points = num_points if num_points is not None else ai_cfg.num_points
    batch_size = batch_size if batch_size is not None else ai_cfg.batch_size
    epochs = epochs if epochs is not None else ai_cfg.epochs
    learning_rate = learning_rate if learning_rate is not None else ai_cfg.learning_rate
    checkpoint_path = checkpoint_path if checkpoint_path is not None else ai_cfg.checkpoint_path
    data_parts = data_parts if data_parts is not None else ai_cfg.parts
    lr_step_size = lr_step_size if lr_step_size is not None else getattr(ai_cfg, "lr_step_size", 10)
    lr_gamma = lr_gamma if lr_gamma is not None else getattr(ai_cfg, "lr_gamma", 0.5)

    # 1. Konfigürasyon Doğrulama
    if num_classes <= 0 or num_points <= 0 or batch_size <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("Tüm hiperparametreler (num_classes, num_points, batch_size, epochs, learning_rate) pozitif olmalıdır.")

    logger.info(
        f"Egitim baslatiliyor -> Epoch: {epochs}, Batch Size: {batch_size}, "
        f"LR: {learning_rate}, Nokta: {num_points}, Parcalar: {data_parts}"
    )

    # Cihaz seçimi (GPU varsa CUDA, yoksa CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Egitim Cihazi: {device}")
    if device.type == "cuda":
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Toplam Bellek: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        except Exception as e:
            logger.warning(f"GPU bilgisi alinirken uyari: {e}")

    # 2. Veri Seti Yükleme ve Bölme
    try:
        full_dataset = DentalMeshDataset(parts=data_parts, num_points=num_points, augment=True)
    except Exception as e:
        logger.error(f"Veri seti yuklenirken hata olustu: {e}")
        return False

    total_count = len(full_dataset)
    if total_count == 0:
        logger.error(f"Belirtilen parcalar ({data_parts}) altinda gecerli vaka bulunamadi!")
        return False

    # Güvenli train / val bölmesi (En az 1 vaka train, 1 vaka val olmalı)
    if total_count == 1:
        logger.warning("Yalnizca 1 vaka mevcut. Hem egitim hem dogrulama icin ayni veri kullanilacak.")
        train_dataset = full_dataset
        val_dataset = full_dataset
    else:
        train_size = max(1, int(0.8 * total_count))
        val_size = total_count - train_size
        if val_size == 0:
            val_size = 1
            train_size = total_count - 1
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    drop_last = len(train_dataset) >= batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Toplam Vaka: {total_count} | Egitim: {len(train_dataset)} | Dogrulama: {len(val_dataset)}")

    # 3. Model, Loss Fonksiyonu ve Optimizer
    try:
        in_channels = getattr(ai_cfg, "in_channels", 6)
        model = DentalPointNetSeg(in_channels=in_channels, num_classes=num_classes).to(device)
    except Exception as e:
        logger.error(f"Model olusturulamadi: {e}")
        return False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    best_val_acc = 0.0
    best_val_miou = 0.0
    start_epoch = 1

    # Devam edilecek checkpoint varsa yukle
    if resume_path:
        if not os.path.exists(resume_path):
            logger.error(f"Devam edilecek checkpoint bulunamadi: {resume_path}")
            return False
        try:
            logger.info(f"Checkpoint yukleniyor: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_miou = ckpt.get("best_val_miou", 0.0)
                best_val_acc = ckpt.get("best_val_acc", 0.0)
                logger.info(
                    f"Egitim {start_epoch}. epoch'tan devam ettiriliyor "
                    f"(Onceki mIoU: %{best_val_miou * 100:.2f}, Acc: %{best_val_acc * 100:.2f})"
                )
            else:
                model.load_state_dict(ckpt)
                logger.info("Salt model agirliklari (state_dict) yuklendi.")
        except Exception as ckpt_err:
            logger.error(f"Checkpoint yuklenemedi: {ckpt_err}")
            return False

    if start_epoch > epochs:
        logger.info(f"Mevcut epoch ({start_epoch - 1}) zaten hedeflenen epoch sayisina ({epochs}) ulasmis.")
        return True

    # 4. Eğitim Döngüsü
    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_train_loss = 0.0
            correct_train_points = 0
            total_train_points = 0
            batch_count = 0

            train_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] - Egitim")
            for batch in train_bar:
                try:
                    points = batch["points"].to(device)   # Boyut: (B, 6, N)
                    labels = batch["labels"].to(device)   # Boyut: (B, N)

                    optimizer.zero_grad()
                    logits = model(points)                # Boyut: (B, NUM_CLASSES, N)

                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()
                    batch_count += 1

                    preds = torch.argmax(logits, dim=1)
                    correct_train_points += (preds == labels).sum().item()
                    total_train_points += labels.numel()

                    train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

                except torch.cuda.OutOfMemoryError:
                    logger.warning("GPU Belleği Yetmedi (CUDA OutOfMemory)! Bellek temizleniyor...")
                    torch.cuda.empty_cache()
                    continue
                except Exception as batch_err:
                    logger.error(f"Batch islenirken hata olustu: {batch_err}")
                    continue

            scheduler.step()
            train_acc = (correct_train_points / total_train_points) if total_train_points > 0 else 0.0
            avg_train_loss = (total_train_loss / batch_count) if batch_count > 0 else 0.0

            # 5. Doğrulama (Validation) Döngüsü
            model.eval()
            total_val_loss = 0.0
            correct_val_points = 0
            total_val_points = 0
            val_batch_count = 0
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    try:
                        points = batch["points"].to(device)
                        labels = batch["labels"].to(device)

                        logits = model(points)
                        loss = criterion(logits, labels)

                        total_val_loss += loss.item()
                        val_batch_count += 1

                        preds = torch.argmax(logits, dim=1)
                        correct_val_points += (preds == labels).sum().item()
                        total_val_points += labels.numel()

                        all_val_preds.append(preds.detach().cpu())
                        all_val_targets.append(labels.detach().cpu())
                    except Exception as val_err:
                        logger.error(f"Dogrulama batch hatasi: {val_err}")
                        continue

            avg_val_loss = (total_val_loss / val_batch_count) if val_batch_count > 0 else 0.0

            if all_val_preds and all_val_targets:
                cat_preds = torch.cat(all_val_preds, dim=0)
                cat_targets = torch.cat(all_val_targets, dim=0)
                val_metrics = evaluate_segmentation_batch(cat_preds, cat_targets, num_classes=num_classes)
                val_acc = val_metrics["accuracy"]
                val_miou = val_metrics["mIoU"]
                val_dice = val_metrics["dice"]
            else:
                val_acc = (correct_val_points / total_val_points) if total_val_points > 0 else 0.0
                val_miou = 0.0
                val_dice = 0.0

            logger.info(
                f"[Epoch {epoch}/{epochs}] "
                f"Egitim Loss: {avg_train_loss:.4f}, Acc: %{train_acc * 100:.2f} | "
                f"Dogrulama Loss: {avg_val_loss:.4f}, Acc: %{val_acc * 100:.2f}, "
                f"mIoU: %{val_miou * 100:.2f}, Dice: %{val_dice * 100:.2f}"
            )

            # En yüksek mIoU veya doğrulamaya ulaşıldığında ağırlıkları güvenli kaydet
            is_better = (val_miou > best_val_miou) or (best_val_miou == 0.0 and val_acc > best_val_acc)
            if is_better:
                best_val_miou = max(best_val_miou, val_miou)
                best_val_acc = max(best_val_acc, val_acc)
                try:
                    checkpoint_dir = os.path.dirname(checkpoint_path)
                    if checkpoint_dir:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_data = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_miou": best_val_miou,
                        "best_val_acc": best_val_acc,
                        "num_classes": num_classes,
                        "num_points": num_points,
                    }
                    torch.save(checkpoint_data, checkpoint_path)
                    logger.info(
                        f"En iyi model agirligi guncellendi -> {checkpoint_path} "
                        f"(Val mIoU: %{val_miou * 100:.2f}, Val Acc: %{val_acc * 100:.2f})"
                    )
                except Exception as save_err:
                    logger.error(f"Checkpoint kaydedilemedi: {save_err}")

        return True

    except KeyboardInterrupt:
        logger.warning("Kullanici tarafindan egitim durduruldu (KeyboardInterrupt).")
        try:
            interrupted_path = checkpoint_path.replace(".pth", "_interrupted.pth")
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_miou": best_val_miou,
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint_data, interrupted_path)
            logger.info(f"Mevcut model durumu kaydedildi: {interrupted_path}")
        except Exception:
            pass
        return False
    except Exception as e:
        logger.error(f"Beklenmeyen egitim hatasi: {e}")
        return False


if __name__ == "__main__":
    parser = get_base_parser(description="Dental 3D Nokta Bulutu Segmentasyon Modeli Egitimi")
    parser.add_argument("--epochs", type=int, default=None, help="Toplam epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch boyutu")
    parser.add_argument("--lr", type=float, default=None, help="Ogrenme orani (Learning rate)")
    parser.add_argument("--num-points", type=int, default=None, help="Model girdi nokta sayisi")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model agirlik dosya yolu (.pth)")
    parser.add_argument("--resume", type=str, default=None, help="Egitime devam edilecek checkpoint dosya yolu (.pth)")
    parser.add_argument("--parts", nargs="+", default=None, help="Egitilecek veri parcalari (orn: data_part_1 data_part_2)")

    args = parser.parse_args()

    # Konfigurasyonu yukle
    cfg = load_config(args.config)

    # CLI argumanlari varsa onceliklendir
    train(
        config=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_points=args.num_points,
        checkpoint_path=args.checkpoint,
        data_parts=args.parts,
        resume_path=args.resume
    )