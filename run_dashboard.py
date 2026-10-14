"""
Launcher script to run the Solar Wind Streamlit Dashboard with a single standard Python command:
    python run_dashboard.py
"""

import sys
import subprocess
import os

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    print("=" * 65)
    print("🚀 Launching Solar Wind Mission Control Dashboard...")
    print(f"Target App: {app_path}")
    print("=" * 65)
    try:
        # Run streamlit command via Python subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch Streamlit: {e}")
        print("Please ensure you have installed the required packages:")
        print("    pip install -r requirements.txt")
