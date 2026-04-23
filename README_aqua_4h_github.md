# AquaFunded BTCUSD 4H cloud bot

This is the no-Mac, cloud-friendly version.

## What it does

- runs every 4 hours in the cloud
- logs into AquaFunded MT5 web terminal
- closes any still-open BTCUSD position from the prior 4H candle
- computes the fresh BTCUSD 4H signal
- places the new market order with fixed SL/TP from the model
- tracks payout-cycle state in `aqua-4h-agent-state.json`
- keeps lot size conservative early, then scales from `0.4` upward only when drawdown and payout rules allow it
- sends Telegram updates if tokens are set

## Files

- `aqua_4h_github_runner.py`
- `aqua-4h-agent-state.json`
- `btc_4h_v1_cloud_runner.py`
- `btc_4h_v1_improved_manual.py`
- `btc_4h_knn_v12_lean.py`
- `requirements-aqua-4h-github.txt`
- `.github/workflows/aqua-4h.yml`

## GitHub setup

Put the workflow file at:

`.github/workflows/aqua-4h.yml`

The local copy is:

`github-actions-aqua-4h.yml`

## Required repo secrets

- `AQUA_MT5_LOGIN`
- `AQUA_MT5_PASSWORD`

Optional:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Recommended:

- `AQUA_INITIAL_BALANCE=100000`
- `AQUA_PAYOUT_DAYS=14`

## Schedule

The workflow uses UTC:

`5 0,4,8,12,16,20 * * *`

That means Nigeria time:

- `01:05`
- `05:05`
- `09:05`
- `13:05`
- `17:05`
- `21:05`

## Important notes

- This depends on the AquaFunded web terminal layout staying roughly the same.
- GitHub Actions cron can run a few minutes late sometimes.
- The workflow commits `aqua-4h-agent-state.json` back into the repo after each run so the risk engine remembers cycle profit, daily profit, and recent trade results.
- If you later move to a VPS, the same runner logic can be reused there.
