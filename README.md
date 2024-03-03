# Utilities for processing EBSD data

This project provides utilities for analysing EBSD data, including determination of GND densities, channelling fractions, and orientation clusters. The code is designed to read in data in CSV files as formatted by Thermo Fisher’s Pathfinder acquisition software, but could be modified to work with data of other formats.

This codebase is currently undergoing a major refactor, the goals of which are to:
- Improve the overall clarity of the code to make it easier to contribute to.
- Add type hints across the codebase to aid in additions and refactors.
- Introduce proper segregation of responsibilities across modules.
- Introduce better data interchange formats between functional components.
- Make most settings configurable via config files.
- Add a central CLI control program from which all analysis functions can be accessed.
- Properly document all functions with respect to equations featured in the accompanying thesis.
- Rename variables and functions for improved clarity and ensure PEP 8 compliance.

This code is provided without any guarantees, as set out in the licence. During the refactoring process, this particularly applies, and there is likely to be code that will not function correctly, if at all. Contributions will not be accepted during this period.

## Contributors:
- O.J. Whiteside, primary author.
- Dr G. Höbler, author of original code to calculate channelling fractions.
