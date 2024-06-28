# To-Dos

- [ ] Swap 1,5 and 5,1

## Gregor
- [ ] Check Jonas code
- [x] Grid-Search CSV in metaheuristics-controller
- [ ] VNS:
  - [ ] Reorder different number of weeks, so profit is maximized
- [ ] Pseudocode
- [ ] Search for papers: Round-robin-tournament (https://en.wikipedia.org/wiki/Round-robin_tournament)

## Jonas 
- [x] Create final file(s)
- [x] Tabu-Search:
  - [x] Only change weeks, which are not changed
  - [x] Save certain week combinations, which are allowed
- [x] Implement Grid-Search for finding optimal parameters
- [x] Check for `NotImplementedErrors`
- [x] Check code
  - [x] helper.py
  - [x] neighborhoods.py
  - [x] validation.py
  - [x] large_neighborhood_search_simulated_annealing.py
  - [x] large_neighborhood_search.py
  - [x] lns_ts_simulated_annealing.py
  - [x] lns_ts.py
  - [x] metaheuristics_controller.py
  - [x] simulated_annealing.py
  - [x] tabu_search.py
- [x] LNS:
  - [x] Reorder weeks
    - [x] Maximum profit (takes too long)
    - [x] At random
  - [x] Adapt weights for selection of method (starting with random -> later the other method(s))
- [x] Disable Multithreading numpy
- [x] Implemenet timeout with CPU-timeout (os.times)
- [ ] Pseudocode
  - [x] `select_random_weeks`
  - [x] `insert_games_random_week`
  - [x] `select_n_worst_weeks`
  - [x] `insert_games_max_profit_per_week`
  - [x] `reorder_week_max_profit`
  - [x] `random_reorder_weeks`
  - [ ] Check for more
- [x] Create submission file with Simulated annealing or so.
- [x] Combination of all
- [ ] Reactive Tabu-Search
- [ ] Select time-consuming operations not at the end of iterations -> Change more
- [ ] Bug fixing of lns_ts_sa! (WhatsApp)
- [ ] Remove print-statements
