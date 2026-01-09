import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
ADAPTER_PATH = "adapters/fine_tuned"
RAG_FILE = "rag_source/oncall_doctor.txt"

DATE_RE = re.compile(r"\bDATE:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\b")

def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")

def split_blocks(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    return [b.strip() for b in raw.split("\n\n") if b.strip()]

def extract_date(block: str) -> Optional[str]:
    m = DATE_RE.search(block)
    return m.group(1) if m else None

def pick_block_for_today(raw: str) -> str:
    today = date.today().isoformat()
    blocks = split_blocks(raw)

    best: Tuple[Optional[str], str] = (None, "")
    for b in blocks:
        d = extract_date(b)
        if d == today:
            return b
        if d and (best[0] is None or d > best[0]):
            best = (d, b)

    return best[1]

def format_doctor_block(block: str) -> str:
    if not block:
        return "Relevant doctor: Not available."
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return "Relevant doctor:\n" + "\n".join(lines)

def strip_mlx_report(text: str) -> str:
    drop_prefixes = ("Prompt:", "Generation:", "Peak memory:")
    kept = [ln for ln in text.splitlines() if not ln.strip().startswith(drop_prefixes)]
    return "\n".join(kept).strip()

def generate_answer(user_query: str, max_tokens: int = 700, temp: float = 0.2) -> str:
    cmd = [
        "mlx_lm.generate",
        "--model", MODEL,
        "--adapter-path", ADAPTER_PATH,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--prompt", user_query,
    ]

    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", MODEL,
            "--adapter-path", ADAPTER_PATH,
            "--max-tokens", str(max_tokens),
            "--temp", str(temp),
            "--prompt", user_query,
        ]
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )

    output = (cp.stdout or "").strip()
    if not output:
        return "Error: no output from mlx_lm.generate"

    return strip_mlx_report(output)

def get_user_query() -> str:
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:]).strip()
        if q:
            return q
    return input("Enter your message: ").strip()

def main() -> None:
    user_query = get_user_query()
    if not user_query:
        print("Error: empty message.")
        return

    raw = read_text(RAG_FILE)
    doctor_block = pick_block_for_today(raw)
    doctor_text = format_doctor_block(doctor_block)

    answer = generate_answer(user_query, max_tokens=700, temp=0.2)

    print(answer.rstrip())
    print()
    print(doctor_text)

if __name__ == "__main__":
    main()