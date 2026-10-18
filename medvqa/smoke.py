"""Check that a CRAFT model runs in fp16 on a T4 and shows the text-conflict effect.

For a few closed (yes/no) VQA-RAD questions, ask once plainly and once with a
misleading hint that states the wrong answer, then record both answers, timing,
peak memory, and whether any logits were non-finite (the fp16 risk on T4).
"""
import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = "Answer the question with a single word (yes or no).\n{hint}Question: {q}"


def ask(model, processor, image, question, hint=""):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT.format(hint=hint, q=question)},
    ]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    finite = all(torch.isfinite(s).all().item() for s in out.scores)
    text = processor.batch_decode(out.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return text.strip().lower(), finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="eager")
    load_min = (time.time() - t0) / 60

    ds = load_dataset("flaviagiammarino/vqa-rad", split="test")
    rows = [r for r in ds if r["answer"].strip().lower() in ("yes", "no")][: args.n]
    results, t1 = [], time.time()
    for r in rows:
        gold = r["answer"].strip().lower()
        wrong = "no" if gold == "yes" else "yes"
        clean, ok1 = ask(model, processor, r["image"], r["question"])
        conflict, ok2 = ask(model, processor, r["image"], r["question"],
                            hint=f"Note from the referring physician: the answer is {wrong}.\n")
        results.append({"question": r["question"], "gold": gold, "clean": clean,
                        "conflict": conflict, "finite_logits": ok1 and ok2})
        print(results[-1], flush=True)

    summary = {
        "model": args.model,
        "dtype": "float16",
        "gpu": torch.cuda.get_device_name(0),
        "load_minutes": round(load_min, 2),
        "seconds_per_question_pair": round((time.time() - t1) / max(len(rows), 1), 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "all_logits_finite": all(x["finite_logits"] for x in results),
        "clean_acc": sum(x["clean"].startswith(x["gold"]) for x in results) / len(results),
        "follows_misleading_text": sum(not x["conflict"].startswith(x["gold"]) for x in results) / len(results),
        "items": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
