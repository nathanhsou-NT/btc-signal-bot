# Automação do retreino semanal — guia de setup

O que foi feito nesta pasta:

- Removi o token do Telegram e o chat id que estavam escritos direto nos
  arquivos (`alert.py`, `alert_ml.py`, `btc_alert-.py`) — agora eles vêm de
  variáveis de ambiente. Isso é obrigatório para poder subir o código pro
  GitHub sem vazar credenciais.
- Criei `.github/workflows/weekly-retrain.yml`: toda segunda-feira às 07:00
  UTC (04:00 na Flórida em horário de verão), o GitHub roda sozinho:
  `fetch_data.py` → `build_dataset.py` → `train_walkforward.py --save-model`,
  te manda um resumo do resultado no Telegram, e sobe o `model.joblib` novo
  pro repositório.
- Criei `start_bot.bat` (só na sua máquina, **não vai pro GitHub**) que dá
  `git pull` — puxando o modelo mais recente — antes de iniciar o
  `alert_ml.py`. `start_bot.example.bat` é o modelo sem o token, caso precise
  recriar.
- Já dei `git init` e o primeiro commit aqui na pasta, sem nenhum segredo no
  histórico (conferi com `git grep`).

Falta só: criar o repositório no GitHub e conectar. Isso eu não consigo fazer
por você (preciso das suas credenciais), mas é rápido:

## 1. Criar o repositório no GitHub

1. Acesse [github.com/new](https://github.com/new).
2. Nome sugerido: `sinais-btc` (marque como **Private**, já que o código
   fala do seu modelo/estratégia).
3. **Não** marque "Add a README" — a pasta já tem tudo.
4. Clique em "Create repository".

## 2. Subir o código

Abra um terminal (PowerShell ou Prompt de Comando) **nesta pasta**
(`C:\Users\Natha\OneDrive\Área de Trabalho\Sinais`) e rode:

```
git remote add origin https://github.com/SEU_USUARIO/sinais-btc.git
git branch -M main
git push -u origin main
```

(troque `SEU_USUARIO` pelo seu usuário do GitHub — o GitHub mostra o comando
exato na tela depois de criar o repo).

## 3. Configurar os Secrets no GitHub

No repositório: **Settings → Secrets and variables → Actions → New
repository secret**. Crie dois:

- `TELEGRAM_TOKEN` → o token do seu bot
- `TELEGRAM_CHAT_ID` → `7913921593`

Sem isso o workflow ainda retreina o modelo normalmente, só não manda a
notificação.

## 4. Testar sem esperar a segunda-feira

Na aba **Actions** do repositório → clique em "Retreino semanal do modelo
BTC" → **Run workflow**. Acompanhe os logs; no final deve aparecer o commit
automático com o `model.joblib` atualizado e a notificação no Telegram (se
configurou os secrets).

## 5. Como o bot local pega o modelo novo

Toda vez que for rodar o bot, use o `start_bot.bat` (dá duplo-clique) em vez
de rodar `python alert_ml.py` direto — ele atualiza o repositório local
(pegando o modelo retreinado) antes de iniciar.

## Recomendação de segurança

O token do Telegram (`8767479244:...`) já esteve escrito em texto puro nos
arquivos por um tempo. Não é urgente, mas o ideal é gerar um token novo no
[@BotFather](https://t.me/BotFather) (`/revoke` no bot atual → `/newtoken` ou
similar) e atualizar em `start_bot.bat` e nos Secrets do GitHub. Assim o
token antigo, que só existiu localmente, deixa de valer.

## Observação sobre OneDrive

Esta pasta é sincronizada pelo OneDrive. Operações de git (init/commit) às
vezes engasgam aqui por causa do sync na nuvem — se algum comando `git`
travar ("Unable to create index.lock"), feche o OneDrive temporariamente
(ícone na bandeja → Pausar sincronização) ou tente de novo depois de alguns
segundos.
