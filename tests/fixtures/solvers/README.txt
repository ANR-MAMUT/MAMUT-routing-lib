Solution files in the exact formats real solvers write, reproduced from each
solver's output code (read on 2026-11-02; the binaries were not run here):

- lkh3-mtsp.sol  LKH-3 MTSP_SOLUTION_FILE (SRC/MTSP_WriteSolution.c, LKH-3.0.6 as
                 in github.com/c4v4/LKH3 5bd6368): "<name>, Cost: <penalty>_<cost>",
                 then one "1 a b 1 (#k)  Cost: x" line per vehicle; node 1 is the depot;
                 empty vehicles appear as "1 1 (#0)  Cost: 0" with MTSP_MIN_SIZE = 0.
- hgs-cvrp.sol   HGS-CVRP (github.com/vidalt/HGS-CVRP 1a92795,
                 Population::exportCVRPLibFormat): "Route #k: ..." and the penalized cost
                 printed with the default 6 significant digits ("Cost 1.23457e+06").
- ails2.sol      AILS-II (INFORMSJoC/2023.0106 4ff3611 + paper7 campaign3/patch_ails2.py,
                 Solution.toString): trailing spaces after each route, Java
                 Double.toString cost ("Cost 1.2345678E7").
- filo2.sol      FILO2 (github.com/acco93/filo2 17b8f84, Solution::store_to_file):
                 std::to_string of a double cost ("Cost 21.000000").

Customer ids are 1..n in all four (the depot is file node 1 / library index 0).
