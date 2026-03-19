import subprocess
import sys


def main():
    # Orchestrator
    outp = subprocess.run(
        [sys.executable, "-m", "ui.gui"],
        check=True,
        text=True,
        capture_output=True,
    )
    if "Done" in outp.stdout:
        subprocess.run(
            [sys.executable, "-m", "ui.frase"],
            check=True,
            text=True,
            capture_output=True,
        )
    else:
        print("generator didn't run successfully :(")


if __name__ == "__main__":
    main()
