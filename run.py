"""
run.py  -  Vibely Movie Recommender
Usage:  C:\ev\Scripts\python.exe run.py
"""
import subprocess, sys, os, time, webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT     = 8500

def get_python():
    ev = r"C:\ev\Scripts\python.exe"
    return ev if os.path.exists(ev) else sys.executable

def main():
    python = get_python()
    print("=" * 45)
    print("  Vibely - Emotion-Based Movie Recommender")
    print("=" * 45)
    print(f"  URL : http://localhost:{PORT}")
    print("=" * 45)

    cmd = [
        python, "-m", "streamlit", "run",
        os.path.join(BASE_DIR, "app.py"),
        "--server.port",          str(PORT),
        "--server.headless",      "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base",           "dark",
        "--theme.primaryColor",   "#7c3aed",
        "--theme.backgroundColor","#0a0a0f",
        "--theme.secondaryBackgroundColor", "#13131f",
        "--theme.textColor",      "#e8e8f0",
    ]

    proc = subprocess.Popen(cmd, cwd=BASE_DIR)
    time.sleep(4)
    webbrowser.open(f"http://localhost:{PORT}")
    print("\nPress Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        proc.terminate()
        print("Bye!")

if __name__ == "__main__":
    main()
