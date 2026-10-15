#!/usr/bin/env python3
"""Download the full-text PDF of every thesis reference into one folder.

Run it in Google Colab (saves straight into Google Drive):

    from google.colab import drive
    drive.mount('/content/drive')
    !python fetch_ref_pdfs.py --out "/content/drive/MyDrive/CAEM refs"

or paste this whole file into a Colab cell after mounting Drive and run it
(it then saves to My Drive/CAEM refs automatically),

or on any computer with Python 3.8+ and internet:

    python fetch_ref_pdfs.py --out "CAEM refs"

No extra packages are needed. Files are named "[NN] Author Year - Title.pdf"
using the reference numbers printed in the thesis. Already-downloaded files
are skipped, so the script can be re-run safely. A report (_download_report.csv)
and a list of papers to fetch by hand (_MISSING.txt) are written to the folder.

How each paper is found, in order:
  1. the known sources listed in REFS (arXiv, ACL Anthology, PMLR, NeurIPS, ...);
  2. arXiv title search;
  3. Semantic Scholar open-access PDF lookup.
An arXiv PDF is only accepted if the arXiv title matches the expected title,
so a wrong ID can never save the wrong paper.
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = "Mozilla/5.0 (thesis reference fetcher; academic use)"

# (thesis number, short name for the file, correct title, [sources])
# source forms: "arxiv:ID", "acl:ID", "url:https://..."
REFS = [
    (1, "Alkaissi 2023", "Artificial Hallucinations in ChatGPT: Implications in Scientific Writing",
     ["url:https://europepmc.org/articles/PMC9939079?pdf=render",
      "url:https://pmc.ncbi.nlm.nih.gov/articles/PMC9939079/pdf/"]),
    (2, "Yang 2024", "Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond", ["arxiv:2304.13712"]),
    (3, "Lin 2022", "TruthfulQA: Measuring How Models Mimic Human Falsehoods", ["acl:2022.acl-long.229", "arxiv:2109.07958"]),
    (4, "Cherian 2024", "Large language model validity via enhanced conformal prediction methods", ["arxiv:2406.09714"]),
    (5, "Mohri 2024", "Language Models with Conformal Factuality Guarantees", ["arxiv:2402.10978"]),
    (6, "Hu 2022", "LoRA: Low-Rank Adaptation of Large Language Models", ["arxiv:2106.09685"]),
    (7, "OpenAI 2023", "GPT-4 Technical Report", ["arxiv:2303.08774"]),
    (8, "Anthropic 2023", "Claude 2", ["url:https://www.anthropic.com/news/claude-2"]),
    (9, "Anil 2023", "PaLM 2 Technical Report", ["arxiv:2305.10403"]),
    (10, "Min 2023", "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation", ["acl:2023.emnlp-main.741", "arxiv:2305.14251"]),
    (11, "Ji 2023", "Survey of Hallucination in Natural Language Generation", ["arxiv:2202.03629"]),
    (12, "Huang 2023", "Large Language Models Can Self-Improve", ["acl:2023.emnlp-main.67", "arxiv:2210.11610"]),
    (13, "Guo 2017", "On Calibration of Modern Neural Networks", ["url:https://proceedings.mlr.press/v70/guo17a/guo17a.pdf", "arxiv:1706.04599"]),
    (14, "Kadavath 2022", "Language Models (Mostly) Know What They Know", ["arxiv:2207.05221"]),
    (15, "Xiong 2023", "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs", ["arxiv:2306.13063"]),
    (16, "Kirkpatrick 2017", "Overcoming catastrophic forgetting in neural networks", ["arxiv:1612.00796", "url:https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114"]),
    (17, "Farquhar 2024", "Detecting hallucinations in large language models using semantic entropy", ["url:https://www.nature.com/articles/s41586-024-07421-0.pdf"]),
    (18, "Lewis 2020", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
     ["url:https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf", "arxiv:2005.11401"]),
    (19, "Wei 2022", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", ["arxiv:2201.11903"]),
    (20, "Wang 2023", "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ["arxiv:2203.11171"]),
    (21, "Manakul 2023", "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models", ["acl:2023.emnlp-main.557", "arxiv:2303.08896"]),
    (22, "Yang 2018", "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", ["acl:D18-1259", "arxiv:1809.09600"]),
    (23, "Vaswani 2017", "Attention Is All You Need", ["arxiv:1706.03762"]),
    (24, "Qwen 2024", "Qwen2.5 Technical Report", ["arxiv:2412.15115"]),
    (25, "Nikitin 2024", "Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities", ["arxiv:2405.20003"]),
    (26, "Tang 2024", "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents", ["acl:2024.emnlp-main.499", "arxiv:2404.10774"]),
    (27, "Dagan 2006", "The PASCAL Recognising Textual Entailment Challenge", []),
    (28, "Liu 2019", "RoBERTa: A Robustly Optimized BERT Pretraining Approach", ["arxiv:1907.11692"]),
    (29, "Reimers 2019", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", ["acl:D19-1410", "arxiv:1908.10084"]),
    (30, "Johnson 2021", "Billion-scale similarity search with GPUs", ["arxiv:1702.08734"]),
    (31, "McCloskey 1989", "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem", []),
    (32, "Chelli 2024", "Hallucination Rates and Reference Accuracy of ChatGPT and Bard for Systematic Reviews: Comparative Analysis",
     ["url:https://www.jmir.org/2024/1/e53164/PDF"]),
    (33, "Chung 2024", "Scaling Instruction-Finetuned Language Models", ["url:https://jmlr.org/papers/volume25/23-0870/23-0870.pdf", "arxiv:2210.11416"]),
    (34, "Thorne 2018", "FEVER: a large-scale dataset for Fact Extraction and VERification", ["acl:N18-1074", "arxiv:1803.05355"]),
    (35, "Yadkori 2024", "Mitigating LLM Hallucinations via Conformal Abstention", ["arxiv:2405.01563"]),
    (36, "Quach 2024", "Conformal Language Modeling", ["arxiv:2306.10193"]),
    (37, "Niculescu-Mizil 2005", "Predicting Good Probabilities With Supervised Learning",
     ["url:https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf"]),
    (38, "Geifman 2017", "Selective Classification for Deep Neural Networks", ["arxiv:1705.08500"]),
    (39, "Gal 2016", "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
     ["url:https://proceedings.mlr.press/v48/gal16.pdf", "arxiv:1506.02142"]),
    (40, "Tian 2023", "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback",
     ["acl:2023.emnlp-main.330", "arxiv:2305.14975"]),
    (41, "Geng 2024", "A Survey of Confidence Estimation and Calibration in Large Language Models", ["acl:2024.naacl-long.366", "arxiv:2311.08298"]),
    (42, "Chen 2024", "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection", ["arxiv:2402.03744"]),
    (43, "Kamath 2020", "Selective Question Answering under Domain Shift", ["acl:2020.acl-main.503", "arxiv:2006.09462"]),
    (44, "Graves 2014", "Neural Turing Machines", ["arxiv:1410.5401"]),
    (45, "Weston 2015", "Memory Networks", ["arxiv:1410.3916"]),
    (46, "Sukhbaatar 2015", "End-To-End Memory Networks", ["arxiv:1503.08895"]),
    (47, "Graves 2016", "Hybrid computing using a neural network with dynamic external memory", []),
    (48, "Das 2024", "Larimar: Large Language Models with Episodic Memory Control", ["arxiv:2403.11901"]),
    (49, "Xu 2025", "A-MEM: Agentic Memory for LLM Agents", ["arxiv:2502.12110"]),
    (50, "Jeong 2024", "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity",
     ["acl:2024.naacl-long.389", "arxiv:2403.14403"]),
    (51, "Mallen 2023", "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories",
     ["acl:2023.acl-long.546", "arxiv:2212.10511"]),
    (52, "Wang 2023b", "Self-Knowledge Guided Retrieval Augmentation for Large Language Models", ["acl:2023.findings-emnlp.691", "arxiv:2310.05002"]),
    (53, "Marina 2025", "LLM-Independent Adaptive RAG: Let the Question Speak for Itself", ["arxiv:2505.04253"]),
    (54, "Asai 2024", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", ["arxiv:2310.11511"]),
    (55, "Baek 2025", "Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval",
     ["acl:2025.findings-naacl.181", "arxiv:2410.13339"]),
    (56, "Biderman 2024", "LoRA Learns Less and Forgets Less", ["arxiv:2405.09673"]),
    (57, "Dettmers 2023", "QLoRA: Efficient Finetuning of Quantized LLMs", ["arxiv:2305.14314"]),
    (58, "Kalajdzievski 2023", "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA", ["arxiv:2312.03732"]),
    (59, "Wang 2023c", "Orthogonal Subspace Learning for Language Model Continual Learning", ["acl:2023.findings-emnlp.715", "arxiv:2310.14152"]),
    (60, "Natarajan 2013", "Learning with Noisy Labels",
     ["url:https://papers.nips.cc/paper/5073-learning-with-noisy-labels.pdf"]),
    (61, "Zelikman 2022", "STaR: Bootstrapping Reasoning With Reasoning", ["arxiv:2203.14465"]),
    (62, "Kamoi 2024", "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs", ["arxiv:2406.01297"]),
    (63, "Gururangan 2020", "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks", ["acl:2020.acl-main.740", "arxiv:2004.10964"]),
    (64, "Hosseini 2024", "V-STaR: Training Verifiers for Self-Taught Reasoners", ["arxiv:2402.06457"]),
    (65, "Song 2024", "Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models", ["arxiv:2412.02674"]),
    (66, "Yi 2025", "Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence", ["arxiv:2510.16657"]),
    (67, "Das 2025", "Retraining with Predicted Hard Labels Provably Increases Model Accuracy", ["arxiv:2406.11206"]),
    (68, "Huang 2025", "Self-Improvement in Language Models: The Sharpening Mechanism", ["arxiv:2412.01951"]),
    (69, "Lang 2024", "Theoretical Analysis of Weak-to-Strong Generalization", ["arxiv:2405.16043"]),
    (70, "Karpukhin 2020", "Dense Passage Retrieval for Open-Domain Question Answering", ["acl:2020.emnlp-main.550", "arxiv:2004.04906"]),
    (71, "Jiang 2023", "Active Retrieval Augmented Generation", ["acl:2023.emnlp-main.495", "arxiv:2305.06983"]),
    (72, "Joshi 2017", "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension", ["acl:P17-1147", "arxiv:1705.03551"]),
    (73, "Kwiatkowski 2019", "Natural Questions: A Benchmark for Question Answering Research", ["acl:Q19-1026"]),
    (74, "Geva 2021", "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies",
     ["acl:2021.tacl-1.21", "arxiv:2101.02235"]),
    (75, "Clark 2018", "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", ["arxiv:1803.05457"]),
    (76, "Hendrycks 2021", "Measuring Massive Multitask Language Understanding", ["arxiv:2009.03300"]),
    (77, "Kuhn 2023", "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation", ["arxiv:2302.09664"]),
    (78, "Sinha 2025 (replacement candidate)", "The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems",
     ["arxiv:2512.15068"]),
    (79, "Li 2023", "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models", ["acl:2023.emnlp-main.397", "arxiv:2305.11747"]),
    (80, "Valentin 2024", "Cost-Effective Hallucination Detection for LLMs", ["arxiv:2407.21424"]),
    (81, "Vashurin 2025", "Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph",
     ["acl:2025.tacl-1.11", "arxiv:2406.15627"]),
    (82, "Mehta 2023", "An Empirical Investigation of the Role of Pre-training in Lifelong Learning",
     ["arxiv:2112.09153"]),
    (83, "Wu 2022", "Pretrained Language Model in Continual Learning: A Comparative Study", []),
    (84, "Talmor 2019", "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge", ["acl:N19-1421", "arxiv:1811.00937"]),
    (85, "Strubell 2019", "Energy and Policy Considerations for Deep Learning in NLP", ["acl:P19-1355", "arxiv:1906.02243"]),
    (86, "Geva 2021 (duplicate of 74)", "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies",
     ["acl:2021.tacl-1.21", "arxiv:2101.02235"]),
    (87, "Holm 1979", "A Simple Sequentially Rejective Multiple Test Procedure", []),
    (88, "McNemar 1947", "Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages", []),
    (89, "Nandakishor 2025", "Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation",
     ["arxiv:2510.01237"]),
    (90, "Gulcehre 2023", "Reinforced Self-Training (ReST) for Language Modeling", ["arxiv:2308.08998"]),
    (91, "Ho 2023", "Large Language Models Are Reasoning Teachers", ["acl:2023.acl-long.830", "arxiv:2212.10071"]),
    (92, "Robbins 1951", "A Stochastic Approximation Method", []),
    (93, "Polyak 1990", "New stochastic approximation type procedures", []),
    (94, "Ovadia 2019", "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift", ["arxiv:1906.02530"]),
    (95, "Thulasidasan 2019", "On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks", ["arxiv:1905.11001"]),
]

# Papers the thesis should probably cite instead of / in addition to a current reference.
EXTRAS = [
    ("02-alt", "Huang L 2023", "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions", ["arxiv:2311.05232"]),
    ("12-alt", "Huang J 2024", "Large Language Models Cannot Self-Correct Reasoning Yet", ["arxiv:2310.01798"]),
    ("19-alt", "Kojima 2022", "Large Language Models are Zero-Shot Reasoners", ["arxiv:2205.11916"]),
]

ARXIV_ID = re.compile(r"arxiv\.org/abs/([^\s/v]+)")


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower().replace("-", " ")).split()


def title_match(a, b):
    return difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio() >= 0.75


def arxiv_query(params):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    data, _ = http_get(url)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(data).findall("a:entry", ns):
        t = " ".join((e.findtext("a:title", "", ns) or "").split())
        m = ARXIV_ID.search(e.findtext("a:id", "", ns) or "")
        if t and m:
            out.append((m.group(1), t))
    time.sleep(3)  # arXiv API etiquette
    return out


def arxiv_pdf(aid, title):
    """Download an arXiv PDF only if its arXiv title matches the expected title."""
    hits = arxiv_query({"id_list": aid})
    if not hits:
        return None, f"arxiv:{aid} not found"
    if not title_match(hits[0][1], title):
        return None, f"arxiv:{aid} is '{hits[0][1]}' (title mismatch, skipped)"
    data, _ = http_get(f"https://arxiv.org/pdf/{aid}")
    return data, f"arxiv:{aid}"


def arxiv_search(title):
    q = " AND ".join(f"ti:{w}" for w in norm(title)[:12] if len(w) > 2)
    for aid, t in arxiv_query({"search_query": q, "max_results": 5}):
        if title_match(t, title):
            data, _ = http_get(f"https://arxiv.org/pdf/{aid}")
            return data, f"arxiv search -> {aid}"
    return None, "arxiv search: no match"


def s2_search(title):
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?"
           + urllib.parse.urlencode({"query": title, "limit": 5, "fields": "title,openAccessPdf"}))
    data, _ = http_get(url)
    for p in json.loads(data).get("data", []):
        pdf = (p.get("openAccessPdf") or {}).get("url")
        if pdf and title_match(p.get("title", ""), title):
            body, _ = http_get(pdf)
            return body, f"semantic scholar -> {pdf}"
    return None, "semantic scholar: no open-access PDF"


def try_source(src, title):
    kind, _, val = src.partition(":")
    if kind == "arxiv":
        return arxiv_pdf(val, title)
    if kind == "acl":
        # verify the Anthology ID points at the expected paper before downloading
        bib, _ = http_get(f"https://aclanthology.org/{val}.bib")
        m = re.search(r'title\s*=\s*"(.*?)"', bib.decode("utf-8", "replace"), re.S)
        if not m or not title_match(re.sub(r"[{}]", "", m.group(1)), title):
            return None, f"acl:{val} title mismatch, skipped"
        data, _ = http_get(f"https://aclanthology.org/{val}.pdf")
        return data, f"acl:{val}"
    data, ctype = http_get(val)
    return data, val


def safe(s):
    return re.sub(r'[\\/:*?"<>|]', "", s)[:120].strip()


def fetch(num, short, title, sources, out, log):
    label = f"{num:02d}" if isinstance(num, int) else num
    base = os.path.join(out, safe(f"[{label}] {short} - {title}"))
    for ext in (".pdf", ".html"):
        if os.path.exists(base + ext):
            print(f"[{label}] already downloaded")
            log.append([label, title, "ok (existing)", base + ext])
            return True
    notes = []
    attempts = [lambda s=s: try_source(s, title) for s in sources]
    attempts += [lambda: arxiv_search(title), lambda: s2_search(title)]
    for attempt in attempts:
        try:
            data, how = attempt()
        except Exception as e:  # network errors, 403s, timeouts
            notes.append(f"error: {e}")
            continue
        if data is None:
            notes.append(how)
            continue
        if data[:5] == b"%PDF-":
            ext = ".pdf"
        elif how.startswith("http") and b"<html" in data[:2000].lower() and "anthropic.com" in how:
            ext = ".html"  # the Claude 2 reference is a blog post, not a PDF
        else:
            notes.append(f"{how}: not a PDF")
            continue
        with open(base + ext, "wb") as f:
            f.write(data)
        print(f"[{label}] OK  via {how}")
        log.append([label, title, "ok", how])
        return True
    print(f"[{label}] MISSING  {title}")
    log.append([label, title, "MISSING", " | ".join(notes)])
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    colab_drive = "/content/drive/MyDrive"
    default_out = os.path.join(colab_drive, "CAEM refs") if os.path.isdir(colab_drive) else "CAEM refs"
    ap.add_argument("--out", default=default_out, help="output folder (default: My Drive/CAEM refs when run in Colab)")
    ap.add_argument("--only", type=int, nargs="*", help="only these reference numbers, e.g. --only 3 5 16")
    ap.add_argument("--no-extras", action="store_true", help="skip the suggested replacement papers")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched and exit")
    # parse_known_args: ignore the '-f kernel.json' Jupyter/Colab adds when the script is pasted into a cell
    args, _ = ap.parse_known_args()

    items = [r for r in REFS if not args.only or r[0] in args.only]
    if not args.no_extras and not args.only:
        items += EXTRAS
    if args.dry_run:
        for num, short, title, sources in items:
            print(num, short, "|", title, "|", sources or "manual / search only")
        return 0

    os.makedirs(args.out, exist_ok=True)
    log = []
    for num, short, title, sources in items:
        fetch(num, short, title, sources, args.out, log)

    with open(os.path.join(args.out, "_download_report.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["ref", "title", "status", "source / notes"]] + log)
    missing = [r for r in log if r[2] == "MISSING"]
    with open(os.path.join(args.out, "_MISSING.txt"), "w", encoding="utf-8") as f:
        f.write("Download these by hand (e.g. via BRAC University library access) and put them in this folder\n"
                "named like the others, e.g. '[31] McCloskey 1989 - ....pdf'.\n\n")
        for r in missing:
            f.write(f"[{r[0]}] {r[1]}\n    tried: {r[3]}\n")
    print(f"\nDone: {len(log) - len(missing)} downloaded, {len(missing)} missing. See _MISSING.txt")
    return 0


if __name__ == "__main__":
    code = main()
    if "ipykernel" not in sys.modules:  # don't raise SystemExit inside a notebook cell
        sys.exit(code)
