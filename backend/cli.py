"""
QuantaAlpha CLI entry.

Commands:
  quantaalpha mine       - run factor mining
  quantaalpha backtest   - run backtest
  quantaalpha ui         - start log Web UI
  quantaalpha health_check - environment health check
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env (prefer project root, fallback to cwd)
_project_root = Path(__file__).resolve().parents[1] #* lokasi cli.py berada di backend/, jadi naik satu level ke project root
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(".env")

import fire  #* ubah dict menjadi cli command
from pipeline.factor_mining import main as mine 
from pipeline.factor_backtest import main as backtest
from app.utils.health_check import health_check
from app.utils.info import collect_info


def app():
    fire.Fire(
        {
            "mine": mine, #* python cli.py mine menjadi pipeleine.factor_mining.main()
            "backtest": backtest,
            "health_check": health_check,
            "collect_info": collect_info,
        }
    )


if __name__ == "__main__":
    app()
