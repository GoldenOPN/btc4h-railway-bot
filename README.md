# Railway BTC 4H Deploy Folder

This folder is the Railway-ready bundle for the BTC/USD 4H bot.

## What Railway should run

- Start command is already handled by the Dockerfile:

```text
python btc_4h_v1_cloud_runner.py
```

## Required Railway variables

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Required Railway cron schedule

Railway cron uses UTC.

To run just after Nigeria 4H candle closes, use:

```text
5 0,4,8,12,16,20 * * *
```

That corresponds to:

- `01:05`
- `05:05`
- `09:05`
- `13:05`
- `17:05`
- `21:05`

in `Africa/Lagos`.

## Important

The service must exit cleanly after each run. This code already does that, which matches Railway cron requirements.
