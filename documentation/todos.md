# To-Dos

- [ ] Swap 1,5 and 5,1

## Gregor
- [ ] Check Jonas code
- [ ] Grid-Search CSV
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
- [ ] Disable Multithreading numpy
- [ ] Implemenet timeout with CPU-timeout (os.times)
- [ ] Reactive Tabu-Search
- [ ] Select time-consuming operations not at the end of iterations -> Change more
- [ ] Combination of all
- [ ] Remove print-statements