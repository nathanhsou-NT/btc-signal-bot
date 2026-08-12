# Weekly retrain automation — setup guide

What was done in this folder:

- Removed the Telegram token and chat id that were hardcoded directly in
  the files (`alert.py`, `alert_ml.py`, `btc_alert-.py`) — they now come from
  environment variables. This is required to push the code to GitHub without
  leaking credentials.
- Created `.github/workflows/weekly-retrain.yml`: every Monday at 07:00 UTC
  (04:00 in Florida during daylight saving time), GitHub runs on its own:
  `fetch_data.py` → `build_dataset.py` → `train_walkforward.py --save-model`,
  sends you a summary of the result on Telegram, and pushes the new
  `model.joblib` to the repository.
- Created `start_bot.bat` (local machine only, **does not go to GitHub**),
  which runs `git pull` — pulling the latest model — before starting
  `alert_ml.py`. `start_bot.example.bat` is the template without the token,
  in case you need to recreate it.
- Ran `git init` and the first commit in this folder, with no secrets in the
  history (verified with `git grep`).

The only thing left: create the repository on GitHub and connect it. I can't
do that for you (I'd need your credentials), but it's quick:

## 1. Create the repository on GitHub

1. Go to [github.com/new](https://github.com/new).
2. Suggested name: `sinais-btc` (mark it **Private**, since the code
   describes your model/strategy).
3. **Don't** check "Add a README" — the folder already has everything.
4. Click "Create repository".

## 2. Push the code

Open a terminal (PowerShell or Command Prompt) **in this folder**
(`C:\Users\Natha\OneDrive\Área de Trabalho\Sinais`) and run:

```
git remote add origin https://github.com/YOUR_USERNAME/sinais-btc.git
git branch -M main
git push -u origin main
```

(replace `YOUR_USERNAME` with your GitHub username — GitHub shows the exact
command on screen after you create the repo).

## 3. Configure the Secrets on GitHub

In the repository: **Settings → Secrets and variables → Actions → New
repository secret**. Create two:

- `TELEGRAM_TOKEN` → your bot's token
- `TELEGRAM_CHAT_ID` → `7913921593`

Without this the workflow still retrains the model normally, it just won't
send the notification.

## 4. Test without waiting for Monday

In the repository's **Actions** tab → click "Retreino semanal do modelo
BTC" → **Run workflow**. Watch the logs; at the end you should see the
automatic commit with the updated `model.joblib` and the Telegram
notification (if you configured the secrets).

## 5. How the local bot picks up the new model

Every time you run the bot, use `start_bot.bat` (double-click it) instead of
running `python alert_ml.py` directly — it updates the local repository
(pulling the retrained model) before starting.

## Security recommendation

The Telegram token (`8767479244:...`) was written in plain text in the
files for a while. It's not urgent, but the ideal move is to generate a new
token on [@BotFather](https://t.me/BotFather) (`/revoke` on the current bot
→ `/newtoken` or similar) and update it in `start_bot.bat` and in the GitHub
Secrets. That way the old token, which only ever existed locally, stops
being valid.

## Note about OneDrive

This folder is synced by OneDrive. Git operations (init/commit) sometimes
stall here because of the cloud sync — if a `git` command hangs ("Unable to
create index.lock"), pause OneDrive temporarily (tray icon → Pause syncing)
or try again after a few seconds.
