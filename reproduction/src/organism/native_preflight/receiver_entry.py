"""Isolated portable-Python entry point for the installed receiver CLI."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/var/lib/organism/data/bootstrap/site-packages")

if __name__ == "__main__":
    from organism.receiver_service import main
    main()
