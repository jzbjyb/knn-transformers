#!/usr/bin/env bash

source openai_keys.sh
model=codex

if [[ $model == 'codex' ]]; then
  curl https://api.openai.com/v1/completions \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer '${test_key} \
    -d '{
    "model": "text-davinci-003",
    "prompt": "Q: The electrolysis of molten magnesium bromide is expected to produce (A) magnesium at the anode and bromine at the cathode (B) magnesium at the cathode and bromine at the anode (C) magnesium at the cathode and oxygen at the anode (D) bromine at the anode and hydrogen at the cathode. A: Lets think step by step.",
    "max_tokens": 10,
    "temperature": 0,
    "logprobs": 0,
    "echo": false,
    "frequency_penalty": 0.0
    }'
elif [[ $model == 'turbo' ]]; then
  curl https://api.openai.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $test_key" \
    -d '{
      "model": "gpt-3.5-turbo",
      "messages": [{"role": "user", "content": "Hello! What is"}],
      "temperature": 0,
      "frequency_penalty": 0.0
    }'
fi
