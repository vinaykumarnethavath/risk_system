import subprocess
import os

def run_ticker(ticker):
    print(f"Running for {ticker}...")
    # main.py now forces UTF-8 reconfigure, but we capture as text.
    # We pass env just to be safe or run_pipeline style.
    result = subprocess.run([r".venv\Scripts\python.exe", "main.py", "--ticker", ticker], capture_output=True, text=True, encoding='utf-8')
    with open(f"output_{ticker.lower()}.txt", "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- ERRORS ---\n")
            f.write(result.stderr)
    print(f"Finished {ticker}. Code: {result.returncode}")

run_ticker("AAPL")
run_ticker("TSLA")
print("Verification runs complete.")
