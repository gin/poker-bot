# poker-bot

### Setup for running with dev.fun's poker-arena-starter-kit
I created an adaptor for this poker-bot to work with the [dev.fun's poker-arena-starter-kit](https://github.com/devfun-org/poker-arena-starter-kit) so that others who are already familiar with their CLI and tooling can easily use this bot's strategies as well.

1. Clone the dev.fun's poker-arena-starter-kit and this repository to the same directory
    ```sh
    mkdir poker
    cd poker
    git clone https://github.com/devfun-org/poker-arena-starter-kit
    git clone https://github.com/gin/poker-bot.git
    ```

2. Copy and paste the adaptor file from the root of this repo to the root of the starter-kit repo
    ```sh
    cp poker-bot/poker-arena-starter-kit-adaptor/poker_bot_strategy_adaptor.py poker-arena-starter-kit/examples/
    ```

3. Add my repository as a dependency in `poker-arena-starter-kit/pyproject.toml`  
For reference, look at [./poker-arena-starter-kit-adaptor/pyproject.toml](./poker-arena-starter-kit-adaptor/pyproject.toml) 
    - At the bottom, add these lines
        ```toml
          [tool.uv.sources]
          poker-bot = { path = "../../poker/poker-bot", editable = true }
        ```
    - In `dependencies` add `"poker-bot"` like this
        ```toml
        dependencies = [
          "httpx>=0.27",
          "python-dotenv>=1.0",
          "treys>=0.1.8",
          "pokerkit>=0.5",
          "poker-bot",
        ]
        ```

4. To run one of this bot's strategies, in the `poker-arena-starter-kit` directory, prepend `POKER_BOT_STRATEGY=<strategy_name>` with the adaptor file as the agent when running `poker-arena-starter-kit`'s `./pokerkit` command like this:
    ```sh
    cd poker-arena-starter-kit
    POKER_BOT_STRATEGY=royal_adaptive ./pokerkit run --agent poker_bot_strategy_adaptor.py --hands 1000 --players 6
    ```
    The strategies are at: [src/poker_bot/strategies/](src/poker_bot/strategies/)  
    The filename is the strategy name.



----
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

### Benchmark
I ran benchmark of different strategies against simple strategy, where each new strategy is an iteration of the one before. It appears that poker playing strategy can be self-improved in a closed-loop with instruction "create a new strategy based on the previous one but ensure the new strategy consistently beats the ones that came before it". `all_in_everytime` and `monte_carlo` strategies are used as edge cases where `simple` (`src/poker_bot/strategies/simple.py`) strategy is used as baseline. The total cost of this self-improvement poker playing experiment including agentically competing with other bots on dev.fun arena until we lost is **$0.68** (12.6M tokens using DeepSeek V4 Flash via OpenRouter):
```
# bb/100 self-improvement progression starting at adaptive strategy
simple                     +2.9  | ██
all_in_everytime         -123.2  | ██████████████████████████████████████████████████ -
monte_carlo               -15.0  | ██████ -
--------------------------------------------------------------------------------
adaptive                  +18.5  | ███████
counter_adaptive          +27.4  | ███████████
profiled_counter_adaptive +27.4  | ███████████
threshold_pressure        +28.0  | ███████████
anti_threshold            +42.9  | █████████████████


# Ranking by bb/100
1. anti_threshold              +42.9
2. threshold_pressure          +28.0
3. profiled_counter_adaptive   +27.4
4. counter_adaptive            +27.4
5. adaptive                    +18.5
6. simple                       +2.9
7. monte_carlo                 -15.0
8. all_in_everytime           -123.2


# Net Chips
simple                     +71,650   | █
all_in_everytime        -3,079,650   | ██████████████████████████████████████████████████ -
monte_carlo                  -375    | -
adaptive                  +462,325   | ███████
counter_adaptive          +684,890   | ███████████
profiled_counter_adaptive +684,890   | ███████████
threshold_pressure        +700,450   | ███████████
anti_threshold          +1,072,698   | █████████████████


# Win Rate
simple                    50.1% | █████████████████████████
all_in_everytime          89.6% | █████████████████████████████████████████████
monte_carlo               40.0% | ████████████████████
adaptive                  55.2% | ████████████████████████████
counter_adaptive          61.0% | ██████████████████████████████
profiled_counter_adaptive 61.0% | ██████████████████████████████
threshold_pressure        61.0% | ██████████████████████████████
anti_threshold            67.1% | ██████████████████████████████████
```

Recreate the benchmark data with:
```sh
$ uv run selfplay --strat simple --hands 50000 --seed 1

  hands       : 50000
  opponent    : simple x1
  wins/losses : 25068/24878  (push: 54)
  net chips   : +71650
  bb/100      : +2.9
  elapsed     : 0.6s  (78792 hands/s)

$ uv run selfplay --strat all_in_everytime --hands 50000 --seed 1

  hands       : 50000
  opponent    : simple x1
  wins/losses : 44776/5091  (push: 133)
  net chips   : -3079650
  bb/100      : -123.2
  elapsed     : 1.3s  (39025 hands/s)

$ uv run selfplay --strat monte_carlo --hands 50 --seed 1

  hands       : 50
  opponent    : simple x1
  wins/losses : 20/30  (push: 0)
  net chips   : -375
  bb/100      : -15.0
  elapsed     : 1.3s  (38 hands/s)

$ uv run selfplay --strat adaptive --hands 50000 --seed 1

  hands       : 50000
  opponent    : simple x1
  wins/losses : 27602/22358  (push: 40)
  net chips   : +462325
  bb/100      : +18.5
  elapsed     : 0.9s  (58645 hands/s)

$ uv run selfplay --strat counter_adaptive --hands 50000 --seed 1

  hands       : 50000
  opponent    : simple x1
  wins/losses : 30477/19487  (push: 36)
  net chips   : +684890
  bb/100      : +27.4
  elapsed     : 0.8s  (62677 hands/s)

$ uv run selfplay --strat profiled_counter_adaptive --hands 50000 --seed 1

  hands       : 50000
  opponent    : simple x1
  wins/losses : 30477/19487  (push: 36)
  net chips   : +684890
  bb/100      : +27.4
  elapsed     : 0.9s  (57313 hands/s)

$ uv run selfplay --strat threshold_pressure --hands 50000 --seed 1

  hands       : 50000
  opponent    : simple x1
  wins/losses : 30477/19487  (push: 36)
  net chips   : +700450
  bb/100      : +28.0
  elapsed     : 0.8s  (61176 hands/s)

$ uv run selfplay --strat anti_threshold --hands 50000 --seed 1

  hands       : 50000
  opponent    : simple x1
  wins/losses : 33567/16380  (push: 53)
  net chips   : +1072698
  bb/100      : +42.9
  elapsed     : 1.1s  (45679 hands/s)
```