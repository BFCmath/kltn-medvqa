"""Convert the official VQA-RAD release into the CSV layout CRAFT's VQA-RAD scripts expect.

Input: a folder with "VQA_RAD Dataset Public.json" and "VQA_RAD Image Folder/" (the OSF release,
mirrored on Kaggle as shashankshekhar1205/vqa-rad-visual-question-answering-radiology).
Output folder:
    vqa_rad_all.csv          every QA pair
    vqa_rad_closed_all.csv   answer_type == CLOSED (CRAFT's starting point)
    VQA_RAD_Image_Folder/    images, same file names as img_name

    python -m medvqa.data.prepare_vqarad --src <release dir> --out <out dir>
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

FIELDS = ["qid", "img_name", "image_organ", "question", "answer", "answer_type",
          "question_type", "phrase_type", "split", "category"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = json.loads((src / "VQA_RAD Dataset Public.json").read_text())
    rows = []
    for r in raw:
        phrase = str(r.get("phrase_type") or "")
        qtype = str(r.get("question_type") or "").strip()
        rows.append({
            "qid": r.get("qid"),
            "img_name": r["image_name"],
            "image_organ": r.get("image_organ"),
            "question": str(r["question"]).strip(),
            "answer": str(r["answer"]).strip(),
            "answer_type": str(r.get("answer_type") or "").strip().upper(),
            "question_type": qtype,
            "phrase_type": phrase,
            "split": "test" if phrase.startswith("test") else "train",
            "category": qtype or "other",
        })

    img_out = out / "VQA_RAD_Image_Folder"
    img_out.mkdir(exist_ok=True)
    names = sorted({r["img_name"] for r in rows})
    for name in names:
        shutil.copy2(src / "VQA_RAD Image Folder" / name, img_out / name)

    closed = [r for r in rows if r["answer_type"] == "CLOSED"]
    write_csv(out / "vqa_rad_all.csv", rows)
    write_csv(out / "vqa_rad_closed_all.csv", closed)
    yes_no = sum(r["answer"].lower() in ("yes", "no") for r in closed)
    print(f"all={len(rows)} closed={len(closed)} images={len(names)} closed yes/no={yes_no}")


if __name__ == "__main__":
    main()
