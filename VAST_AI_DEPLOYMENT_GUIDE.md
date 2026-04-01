# CAEM Cloud Deployment Guide (Vast.ai)

> **Purpose:** A complete, step-by-step tutorial for moving the CAEM repository from a local Windows machine to a rented cloud GPU (RTX 4090) on Vast.ai for the final 14-hour thesis experiment.

---

## Phase 1: Billing Setup
Because Vast.ai is a cloud marketplace, you need to load a small deposit (typically $10 or $20 USD) before renting a machine.
1. Go to the Billing page.
2. Under "Add Credit", choose **Stripe** (ignore Crypto/Coinbase/BitPay). 
3. Enter a standard Visa or Mastercard that supports international (USD) transactions. 

## Phase 2: Choosing the Template & Fixing Disk Space (CRITICAL)
Before selecting an RTX 4090, you must configure the software environments and storage restrictions.
1. On the "Search" page, locate the **"Edit Image & Config"** button (or "Selected Template" box) at the top left.
2. In the pop-up menu, select the official **PyTorch** template from the Recommended list.
3. **CRITICAL STEP:** Scroll to the very bottom of that pop-up window. Locate the **"Allocated Disk Space"** slider. The default will be a dangerously low 16 GB. **Slide it to 50.00 GB.** 
   *(Failing to do this will cause the 14-hour experiment to crash with a `Disk OutOfSpace` error right as Cycle 1 finishes).*
4. Click **Select & Save**.

## Phase 3: Hardware Selection
1. In the left-hand Search Filters panel, verify these boxes are checked:
   * **Verified** (to filter out unstable amateur hosts)
   * 1x GPU
   * RTX 4090 
2. Wait for the cards on the right to refresh. Look for a machine that costs around **$0.34/hr** with a Reliability percentage above **99.5%**.
3. Click the blue **RENT** button.

## Phase 4: Connection Options
Once your instance switches from "Loading" to "Running" in your "Instances" tab, connect to it using one of two methods:

### Option A: VS Code Remote SSH (Highly Recommended)
Since you write your code in Visual Studio Code, this is the most seamless method.
1. On your local laptop, open VS Code and install the **"Remote - SSH"** extension.
2. Click the blue `>_ CONNECT` button next to your running Vast.ai instance to reveal your SSH command (e.g., `ssh -p 24501 root@12.34.56.78`).
3. In VS Code, click the green `><` icon in the bottom-left corner.
4. Select *Connect to Host*, then *Add New SSH Host*, and paste the exact SSH command.
5. You now have a VS Code window connected directly to the supercomputer. You can edit code, open terminals, and visually drag/drop files.

### Option B: Jupyter Notebook (Web Browser)
If you don't want to use VS Code, click the **Jupyter button** right next to the "CONNECT" button on the Vast.ai interface. It will pop open a familiar web-based graphical file explorer where you can upload files and launch a terminal tab.

## Phase 5: File Transfer and Execution
With your terminal open on the remote server:
1. Clone your project code: `git clone <your-repo-link>`
2. Upload `data/passage_index/` and `outputs/cold_start_memory/` from your laptop using VS Code drag-and-drop, `scp`, or by downloading from Google Drive/HuggingFace directly into the server.
3. Install dependencies: `pip install torch transformers datasets faiss-cpu sentence-transformers`
4. Use `tmux` to prevent SSH disconnects from crashing your script:
   `tmux new -s caem`
5. Run the orchestrator: 
   `python scripts/run_experiment.py --n_questions 5000`
