"""Evaluasi model YOLO pada data beranotasi yang tidak dipakai untuk training (FR-09).

Metrik dihitung oleh Ultralytics dari ground-truth di dataset (file label), bukan dari
gambar biasa tanpa anotasi.

Contoh pemakaian (Windows PowerShell, dari folder proyek):

    python evaluate.py --data "D:\\path\\ke\\dataset\\data.yaml" --split test

Keluaran, di folder evaluation/<nama_run>/:
    metrics_summary.csv     precision, recall, mAP@0.5, mAP@0.5:0.95, kecepatan per gambar
    metrics_per_class.csv   metrik yang sama untuk tiap kelas
    run_info.json           model, data, split, parameter, dan versi library (reproducibility)
    confusion_matrix*.png   confusion matrix
    PR/P/R/F1 curve + contoh prediksi, dibuat otomatis oleh Ultralytics
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation"

SUMMARY_FIELDS = ["precision", "recall", "mAP50", "mAP50_95",
                  "speed_preprocess_ms", "speed_inference_ms", "speed_postprocess_ms"]
CLASS_FIELDS = ["class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluasi model YOLO (Precision, Recall, mAP, confusion matrix) pada data beranotasi."
    )
    parser.add_argument("--data", required=True, type=Path,
                        help="Path ke data.yaml dataset (harus berisi split yang dievaluasi).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help=f"Path bobot model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="Split yang dievaluasi (default: test, yang tidak dipakai saat training).")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Ukuran gambar. Jika dikosongkan, memakai ukuran dari konfigurasi training model.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence minimum untuk evaluasi mAP (default 0.001, standar Ultralytics).")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold untuk NMS (default 0.6).")
    parser.add_argument("--device", default="auto", help='"auto", "cpu", atau nomor GPU seperti "0".')
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Folder induk keluaran.")
    parser.add_argument("--name", default=None, help="Nama run (default: <nama_model>_<split>).")
    return parser.parse_args(argv)


def resolve_device(requested):
    if requested != "auto":
        return requested
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def check_inputs(args):
    if not args.model.is_file():
        sys.exit(f"ERROR: file model tidak ditemukan: {args.model}")
    if not args.data.is_file():
        sys.exit(f"ERROR: file data.yaml tidak ditemukan: {args.data}")
    import yaml
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get(args.split):
        sys.exit(f"ERROR: data.yaml tidak memiliki entri '{args.split}'. "
                 "Tambahkan path split tersebut (gambar + label beranotasi).")


def summarize(metrics):
    """Metrik keseluruhan dari objek hasil model.val()."""
    box = metrics.box
    speed = getattr(metrics, "speed", None) or {}
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "speed_preprocess_ms": float(speed.get("preprocess", float("nan"))),
        "speed_inference_ms": float(speed.get("inference", float("nan"))),
        "speed_postprocess_ms": float(speed.get("postprocess", float("nan"))),
    }


def per_class_rows(metrics):
    """Metrik per kelas. Kelas tanpa instance di data evaluasi tidak muncul (tidak bisa dihitung)."""
    box = metrics.box
    rows = []
    for position, class_index in enumerate(box.ap_class_index):
        precision, recall, ap50, ap = box.class_result(position)
        class_index = int(class_index)
        rows.append({
            "class_id": class_index,
            "class_name": metrics.names[class_index],
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:.4f}" if isinstance(value, float) else value)
                             for key, value in row.items()})


def print_report(summary, rows):
    print("\n=== Ringkasan evaluasi ===")
    print(f"Precision      : {summary['precision']:.4f}")
    print(f"Recall         : {summary['recall']:.4f}")
    print(f"mAP@0.5        : {summary['mAP50']:.4f}")
    print(f"mAP@0.5:0.95   : {summary['mAP50_95']:.4f}")
    print(f"Inferensi/gambar: {summary['speed_inference_ms']:.1f} ms")
    print("\n=== Per kelas ===")
    print(f"{'kelas':<10}{'precision':>10}{'recall':>10}{'mAP@0.5':>10}{'mAP@.5:.95':>12}")
    for row in rows:
        print(f"{row['class_name']:<10}{row['precision']:>10.4f}{row['recall']:>10.4f}"
              f"{row['mAP50']:>10.4f}{row['mAP50_95']:>12.4f}")


def main(argv=None):
    args = parse_args(argv)
    check_inputs(args)
    device = resolve_device(args.device)

    if args.split == "val":
        print("CATATAN: split 'val' dipakai Ultralytics untuk memilih best.pt saat training. "
              "Untuk laporan performa akhir, gunakan --split test.\n")

    import torch
    import ultralytics
    from ultralytics import YOLO

    name = args.name or f"{args.model.stem}_{args.split}"
    val_kwargs = dict(
        data=str(args.data), split=args.split, batch=args.batch, conf=args.conf, iou=args.iou,
        device=device, plots=True, project=str(args.output), name=name, exist_ok=True,
    )
    if args.imgsz:
        val_kwargs["imgsz"] = args.imgsz

    print(f"Model  : {args.model}\nData   : {args.data} (split: {args.split})\nDevice : {device}")
    try:
        metrics = YOLO(str(args.model)).val(**val_kwargs)
    except Exception as error:  # pesan jelas, bukan traceback panjang
        sys.exit(f"ERROR: evaluasi gagal: {type(error).__name__}: {error}")

    save_dir = Path(metrics.save_dir)
    summary = summarize(metrics)
    rows = per_class_rows(metrics)
    write_csv(save_dir / "metrics_summary.csv", [summary], SUMMARY_FIELDS)
    write_csv(save_dir / "metrics_per_class.csv", rows, CLASS_FIELDS)
    (save_dir / "run_info.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz or "dari konfigurasi training model",
        "batch": args.batch, "conf": args.conf, "iou": args.iou, "device": device,
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(summary, rows)
    print(f"\nHasil lengkap (CSV, confusion matrix, grafik) disimpan di:\n{save_dir}")


if __name__ == "__main__":
    main()
