import glob
import os

models = [
    'qwen3.8-9b-distill',
    'prism-ml/bonsai-27b',
    'qwen/qwen3-coder-30b',
    'qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp'
]

for model in models:
    safe = model.replace('/', '-').replace('.', '-')
    files = glob.glob(f'results/*{safe}*best.md')
    if files:
        f = max(files, key=os.path.getmtime)
        with open(f) as fp:
            text = fp.read()
        print(f"=== {model} ({f}) ===")
        for line in text.split('\n')[:80]:
            if any(k in line.lower() for k in ['context', 'gen tok', 'vram', 'gpu', 'offload', 'flash', 'batch', 'verdict', 'elapsed', 'quality', 'score']):
                print(f'  {line.strip()}')
        print()
