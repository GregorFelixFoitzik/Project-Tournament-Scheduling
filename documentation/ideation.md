# Ideation

## Tabu-Search:
- Tabu List: Complete solutions
- Tabu list $\leadsto$ empty neighborhood: Clear tabu-list or restart search


## Simulated annealing


## Combination of all
Combine all three algorithms and use the LNS as *main* component. Apply the Tabu-list, when the possible weekly combinations are created. Additionally, search for literature to use the saturation parameter of Simulated-annealing.

- Genetic algorithms for timetabling: https://ieeexplore.ieee.org/abstract/document/1225396
- Genetic algorithms for timetabling in schools (maybe re-implement this approach): https://d1wqtxts1xzle7.cloudfront.net/33105221/download-libre.pdf?1393640937=&response-content-disposition=inline%3B+filename%3DSUBMITTED_TO_COMPUTATIONAL_OPTIMIZATION.pdf&Expires=1718626324&Signature=gLjz1KM0LdKVsnh9NyPpbCo6F3mziPeGvmTxsJrgoyG5ZqFybXEHp58jF9q-xrp3u6u~qbSULvsdn-ricWPXw1V63XO~EYarhKyNUa3dnR0QK~zcKHcKsuS0aPbxP~~7~qbrDw2NT4rLLHPAXOZ0qUVFs1-MjCP~iGRoDkidcGdTugRTnYbazzKEb~96a-h4cbgW1tip7GDuP~L4p99okxruBQc6w18lO8MHsc2PdYTrdXC1UAFigQrilJujFGa5ta~oJYjB8TBXhrOGLCn6AN4LgbN~t1rCPQpDnnXiCWtofPLdyQkIZ5V9KCKoZhm1Iba4T59HibFnaJ403Nc83A__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

## Genetic algorithms
1. Generate $n$ random/good solutions (depending on the generation speed) (all feasible)
2. Crossover:
   1. Take biggest solutions and swap (single point cross over)
   3. Fix solution: 
      1. Remove all infeasible ones
      2. Add games that are missing
      3. Insert games randomly (due computation time) w.r.t. constrainsts