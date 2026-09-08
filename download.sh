#!/usr/bin/env bash
# download.sh <hf-id>  —  fetch a model's weights into ./models/<id> via hf, so a recipe can serve it
# offline as  model: /models/<id>. Self-contained: bootstraps its own venv + huggingface_hub.
#
#   ./download.sh Qwen/Qwen3-0.6B
#   ./download.sh deepseek-ai/DeepSeek-V4-Flash-0731
#
set -euo pipefail
cd "$(dirname "$0")"

M="${1:-}"
[ -n "$M" ] || { echo "usage: ./download.sh <hf-id>   e.g. ./download.sh Qwen/Qwen3-0.6B"; exit 1; }

# venv — bootstraps hf here, not on you
V=.venv
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/python" -m pip install -q -U pip "huggingface_hub[cli]" >/dev/null

# --- Hugging Face access ------------------------------------------------------------------------------
# Anonymous downloads are rate-limited, and GATED repos (license-agreement models — the uncensored variants, most
# fine-tunes of gated bases) refuse anonymous access outright: the download stalls, then dies with 401 after a while.
# Token order: $HF_TOKEN → ~/.cache/huggingface/token (from `hf auth login`) → ask, when interactive. For a gated
# repo the token must also have been GRANTED access (the agreement is per model) — checked before anything downloads.
# Nothing is stored by this script; `hf auth login` is how a token is kept.
hf_access() {  # hf_access <hf-repo>   — exports HF_TOKEN when one is found or entered; returns 1 when the download cannot work
  local repo="$1" meta gated code who
  if [ -z "${HF_TOKEN:-}" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"; export HF_TOKEN
  fi
  meta="$(curl -s --max-time 15 "https://huggingface.co/api/models/$repo" || true)"
  case "$meta" in
    *'"gated":"auto"'*|*'"gated":"manual"'*|*'"gated":true'*) gated=yes ;;
    *'"gated":false'*) gated=no ;;
    "") echo "  (huggingface.co not reachable — skipping the access check)"; return 0 ;;
    *) echo "✗ $repo: not found on Hugging Face (or private)"; return 1 ;;
  esac
  if [ -z "${HF_TOKEN:-}" ]; then
    if [ "$gated" = yes ]; then
      echo "· $repo is a GATED model — Hugging Face only serves it to an account that accepted its agreement:"
      echo "    1. open https://huggingface.co/$repo and accept the agreement (some repos approve by hand — wait for the mail)"
      echo "    2. create a READ token at https://huggingface.co/settings/tokens"
      echo "    3. export HF_TOKEN=hf_...   or   hf auth login   (keeps it in ~/.cache/huggingface), then rerun"
      if [ -t 0 ]; then
        read -rsp "  paste the token now to continue (input hidden; Enter aborts): " HF_TOKEN; echo
        [ -n "$HF_TOKEN" ] || return 1
        export HF_TOKEN
      else
        return 1
      fi
    else
      echo "· no Hugging Face token (HF_TOKEN unset, no hf auth login) — anonymous downloads are rate-limited; a free READ token"
      echo "  from https://huggingface.co/settings/tokens is faster:  export HF_TOKEN=hf_...   or   hf auth login"
      if [ -t 0 ]; then
        read -rsp "  paste a token to use it now, or Enter to continue anonymously: " HF_TOKEN; echo
        if [ -n "$HF_TOKEN" ]; then export HF_TOKEN; else unset HF_TOKEN; fi
      fi
      [ -n "${HF_TOKEN:-}" ] || return 0
    fi
  fi
  who="$(curl -s --max-time 15 -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 | sed -n 's/.*"name":"\([^"]*\)".*/\1/p' | head -1)"
  [ -n "$who" ] || { echo "✗ the Hugging Face token is not valid (whoami failed) — check HF_TOKEN / hf auth login"; return 1; }
  code="$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 30 -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/$repo/resolve/main/config.json")"
  case "$code" in
    200) if [ "$gated" = yes ]; then echo "· Hugging Face: $who — access to the gated $repo granted"; else echo "· Hugging Face: $who"; fi ;;
    401|403) echo "✗ $who has no access to $repo yet — accept the agreement at https://huggingface.co/$repo (manual approval takes a while), then rerun"; return 1 ;;
    *) echo "  (access check returned HTTP $code — continuing)" ;;
  esac
}
# --------------------------------------------------------------------------------------------------------

hf_access "$M" || exit 1
echo "· downloading $M  →  ./models/$M"
"$V/bin/hf" download --local-dir "models/$M" "$M"
echo "✓ ./models/$M   —   serve it in a recipe as   model: /models/$M"
