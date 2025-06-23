#!/bin/bash

# Завантажуємо змінні з .env
source .env

# Встановлюємо URL з токеном
git remote set-url origin https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$REPO_NAME

# Виконуємо push
git push -u origin main
