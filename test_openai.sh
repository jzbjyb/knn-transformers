#!/usr/bin/env bash

source openai_keys.sh
key="${keys[0]}"
model=turbo

if [[ $model == 'codex' ]]; then
  curl https://api.openai.com/v1/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer '${key} \
    -d '{
    "model": "code-davinci-002",
    "prompt": [0,1,2,3],
    "max_tokens": 0,
    "temperature": 0,
    "logprobs": 0,
    "echo": true,
    "frequency_penalty": 0.0
    }'
elif [[ $model == 'turbo' ]]; then
  curl https://api.openai.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $key" \
    -d '{
      "model": "gpt-3.5-turbo",
      "messages": [{"role": "user", "content": "Hello!"}],
      "temperature": 0,
      "frequency_penalty": 0.0
    }'
fi
