#!/usr/bin/env bash

shell_single_quote() {
  local value="${1-}"
  value=${value//\'/\'\"\'\"\'}
  printf "'%s'" "${value}"
}
