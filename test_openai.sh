#!/usr/bin/env bash

source openai_keys.sh
key="${keys[0]}"

curl https://api.openai.com/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer '${key} \
  -d '{
  "model": "code-davinci-002",
  "prompt": "What is water? What is water? What is water?",
  "max_tokens": 128,
  "temperature": 0,
  "logprobs": 0,
  "echo": false,
  "frequency_penalty": 1.0
}'
