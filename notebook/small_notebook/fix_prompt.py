import re, datasets
import hashlib


prompt_template = r"""Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. Do not wrap $Answer with \boxed{}.

current question: {{question}}

Below are two examples for format reference.
Example question 1: Solve for x: 3x - 5 = 16.

Response:
Add 5 to both sides: 3x = 21.
Divide both sides by 3: x = 7.
Answer: 7

Example question 2: A jacket costs $80 and is on sale for 25% off. What is the sale price?

Response:
25% of 80 is 0.25 × 80 = 20.
Subtract the discount from the original price: 80 − 20 = 60.
Answer: 60

Solve the current question. Remember to put your answer on its own line after "Answer:".
"""
def _hash_prompt_batch(batch):
    hashes = []
    for p in batch["prompt"]:
        try:
            content = p[0]["content"]
        except Exception:
            content = None
        s = "" if content is None else str(content)
        hashes.append(hashlib.sha1(s.encode("utf-8")).hexdigest())
    return {"__prompt_hash__": hashes}

def process_batch(batch):
    prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    suffix = '\n\nRemember to put your answer on its own line after "Answer:".'

    pattern = re.compile(
        re.escape(prefix) + r"(.*?)" + re.escape(suffix),
        re.DOTALL
    )
    prompts_out = []
    questions = []

    for p in batch["prompt"]:
        # p is a list of messages; we modify only the first message's "content"
        msg0 = dict(p[0])  # shallow copy of the first message dict
        content = msg0.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        m = pattern.match(content)
        if m:
            q = m.group(1).strip()
            new_content = prompt_template.replace("{{question}}", q)
        else:
            # If it doesn't match the expected shape, keep content unchanged (safe fallback)
            # and still expose a best-effort "question" (here: the whole content).
            q = content.strip()
            new_content = content  # do not alter non-matching rows

        msg0["content"] = new_content

        # rebuild the prompt list with the updated first message
        new_prompt = list(p)
        new_prompt[0] = msg0

        prompts_out.append(new_prompt)
        questions.append(q)

    return {"prompt": prompts_out, "question": questions}


dataframe = datasets.load_dataset("parquet", data_files=["data/dapo-math-17k.parquet"])["train"]
# 1) Add a temporary hash column over prompt content only

ds_h = dataframe.map(
    _hash_prompt_batch,
    batched=True,
    batch_size=1000,
    num_proc=128
)

# 2) Get first index for each hash
first_index = {}
for idx, h in enumerate(ds_h["__prompt_hash__"]):
    if h not in first_index:
        first_index[h] = idx

# 3) Keep only those first occurrences
keep_indices = sorted(first_index.values())
ds_unique = ds_h.select(keep_indices)

# 4) Drop the temporary column
ds_unique = ds_unique.remove_columns(["__prompt_hash__"])
ds_unique.to_parquet("data/nodup-dapo-math-17k.parquet")

dataframe = datasets.load_dataset("parquet", data_files=["data/nodup-dapo-math-17k.parquet"])["train"]
dataframe_new = dataframe.map(process_batch, batched=True, batch_size=1000, num_proc=128, desc="Normalize prompts")
dataframe_new.to_parquet("data/fixprompt-nodup-dapo-math-17k.parquet")

in_path = "data/aime-2024.parquet"
out_path = "data/fixprompt-aime-2024.parquet"
# Load, transform, save
ds = datasets.load_dataset("parquet", data_files=[in_path])["train"]
ds_new = ds.map(process_batch, batched=True, batch_size=1000, num_proc=128, desc="Normalize prompts")
ds_new.to_parquet(out_path)
print(ds_new)
print(f"Saved to {out_path}")