# EU Energy Dashboard (local Streamlit) — ENTSO-E Transparency Platform

A local Streamlit dashboard that pulls:
- **Day-ahead prices** (ENTSO-E: Market → EnergyPrices)
- **Actual generation per production type** + optional **installed capacity** (ENTSO-E: Generation)

It uses the **`entsoe-apy`** Python package, which expects an API key in `ENTSOE_API` by default.

---

## 1) Setup

Create a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
# mac/linux:
source .venv/bin/activate
# windows (powershell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## 2) Put your API key

### Option A — Environment variable (recommended)
macOS / Linux (bash/zsh):

```bash
export ENTSOE_API="your-security-token-here"
```

Windows (PowerShell):

```powershell
setx ENTSOE_API "your-security-token-here"
# reopen the terminal after setx
```

### Option B — `.env` file (local only)
Copy `.env.template` → `.env` and paste your token. The app loads it via `python-dotenv`.

### Option C — Streamlit secrets (local only)
Copy `.streamlit/secrets.toml.template` → `.streamlit/secrets.toml` and paste your token.
The app will automatically call `entsoe.config.set_config(security_token=...)`.

---

## 3) Run

```bash
streamlit run app.py
```

---

## Notes & gotchas
- Time handling: the app assumes your selected dates are **Europe/Brussels** and converts to UTC for ENTSO-E periods.
- Some endpoints have ENTSO-E limitations (e.g., per-unit generation max interval). `entsoe-apy` will split big windows automatically where supported.
- If you get intermittent errors, ENTSO-E occasionally rate-limits; the library has retries/backoff built in.

---

## References
- `entsoe-apy` docs (API key + configuration): https://entsoe-apy.berrisch.biz/
