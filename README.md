# poker-bot

### Setup
1. Create account from https://dev.fun  
  ```sh
  curl --json '{"handle": "YOUR_HANDLE", "name": "YOUR_NAME", "quote": "BIO_DESCRIPTION"}' https://arena.dev.fun/api/arena/auth/register
  ```
2. Copy `.arena-credentials.example` to `.arena-credentials`
3. Update API key, API prefix, competition ID, and Agent ID in `.arena-credentials`
4. Sync dependencies  
  `uv sync`
5. Run test to ensure no errors  
  `uv run pytest`
6. Run the bot  
  `uv run main.py`

### Test
```sh
uv run pytest

# Linting and formatting
uv run ruff check .

# Type check
uv run ty check
```

### Simulator
Run the terminal poker simulator against the bot logic:

```sh
# Human play against a bot using baseline strategy
uv run simulator.py

# Human play against a bot using strategy in src/poker_bot/strategies/all_in_everytime.py
uv run simulator --strat all_in_everytime

# Eval bot using strat against baseline strat
uv run selfplay --strat all_in_everytime --seed 1
```
