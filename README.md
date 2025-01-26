# Utilities for processing EBSD data

This project provides utilities for analysing EBSD data, including determination of GND densities, channelling fractions, and orientation clusters. The code is designed to read in data in CSV files as formatted by Thermo Fisher’s Pathfinder acquisition software, but could be modified to work with data of other formats.

- [Analysing data](#analysing-data)
- [Other functionality](#other-functionality)
- [Modifying the codebase](#modifying-the-codebase)
- [Config options](#config-options)
- [Contributors](#contributors)
- [Related publications](#related-publications)
- [Recommended citation format](#recommended-citation-format)
- [References](#references)

## Analysing data

The following command will analyse data files chosen from the data directory using an interactive file selector:

```shell
python main.py analyse
```

Alternatively, any data file on disk can be specified using the `-f`/`--file` flag:

```shell
python main.py analyse -f data/ebsd/2205031525.csv
```

For each data file, a CSV file roughly matching the original file in format will be created in the analyses output directory. PNG maps of the data will be created in the map output directory. Prior to analysis, the pixel size of the EBSD data must be set via the associated [config option](#data-options).

### Analysis modes

The content of the generated outputs depends on the analysis modes specified in the config file. Each mode can be enabled or disabled by toggling the associated [config option](#analysis-options).

> [!NOTE]
> All data and maps are generated lazily, so disabling unneeded analysis modes will speed up the analysis process by skipping them.

#### GND densities

This option adds a column of logarithmic GND densities to the analysis and generates a GND density map. Densities are calculated using the entrywise one-norm of the Nye tensor [[1]](#references). This is not physically accurate for systems other than primitive cubic, but serves as a lower bound approximation for others, and is directly proportional to the correct value in face-centred cubic systems. See the associated [config options](#dislocation-density-options) for parameters.

#### Channelling fractions

This option adds a column of ion channelling fractions to the analysis and generates a channelling fraction map. Channelling fractions are generated using a Monte Carlo simulation [[2]](#references). Original simulation code written by Dr G. Hobler. See the associated [config options](#channelling-fraction-options) for parameters.

#### Orientation clusters

This option adds two columns for clustering categories and cluster numbers to the analysis and generates an orientation cluster map. Clusters are determined using an implementation of the DBSCAN clustering algorithm [[3]](#references). Where available, this code is run on a CUDA-compatible GPU. Additionally, a cluster summary section is added to the analysis which lists the average values per cluster for Euler angles (using rotation matrix averaging), pattern qualities, index qualities, GND densities (if enabled), and channelling fractions (if enabled). See the associated [config options](#orientation-clustering-options) for parameters.

#### Resolution reduction

This option reduces the resolution of the EBSD data prior to analysis by kernel averaging using a 2×2 kernel. Can be applied repeatedly for further reduction. This is useful for speeding up analyses when performing orientation clustering on large datasets, due to the quadratic time complexity of the DBSCAN algorithm. Original data files remain unmodified. See the associated [config options](#resolution-reduction-options) for parameters.

> [!IMPORTANT]  
> Resolution reduction can have significant effects on calculated GND densities due to effective conversion of GNDs to SSDs. GND densities have been shown to be inversely proportional to step size [[4]](#references).

## Other functionality

### Add new phases

The following command will begin an interactive dialogue that adds new crystal phases for analysing EBSD data based on information entered by the user:

```shell
python main.py add_phase
```

Phases are referenced by their IDs as per the Pathfinder phase database. If the program finds a database file, information will be extracted from it where possible, otherwise manual data entry is required. Supplementary information not available in the database must always be entered manually. Added phases are stored as JSON files and automatically used when analysing EBSD data.

> [!NOTE]  
> If the necessary phase data is not available when analysing EBSD data, the user is automatically given the opportunity to add the new phase in order to continue with the analysis. As such, it is never necessary to add phases explicitly.

### Run tests

The following command runs acceptance tests that should pass before any changes are merged to this repository:

```shell
python main.py test
```

Tests are run for each file in the test data directory. The tests can also be run for an individual file only using the `-f`/`--file` flag, but that file must be in the test data directory:

```shell
python main.py test -f tests/data/2205031525.csv
```

## Modifying the codebase

Contributions to this codebase are strongly encouraged, in the spirit of open research. If you make any additions or improvements, we strongly encourage you to open a pull request so that others can benefit from your innovations.

### Using different data file formats

Though the codebase is designed to work with the CSV data generated by Pathfinder, it could be easily adapted to process EBSD data generated by other sources. All functions for reading data from and writing outputs to files are contained in the [`filestore`](src/utilities/filestore.py) module. Notably, the `load_from_data` function returns an [`Analysis`](src/data_structures/analysis.py) object. If this method is modified to correctly construct the object from a different data file, the rest of the code should run smoothly.

### Working with non-cubic phases

This codebase was originally written to analyse data from materials with cubic crystal phases. As such, several functions are not implemented for other crystal symmetries, and will raise a `SymmetryNotImplementedError` when called. These functions use `match`/`case` statements to select the correct implementation for the provided symmetry, represented by a [`CrystalFamily`](src/data_structures/phase.py) or [`BravaisLattice`](src/data_structures/phase.py) enum variable. By adding new cases to these statements, data for non-cubic phases can be analysed.

### Adding new analysis modes

Data is analysed by constructing new [`Field`](src/data_structures/field.py), [`Aggregate`](src/data_structures/aggregate.py), or [`Map`](src/data_structures/map.py) objects, and making them accessible using the [`FieldManager`](src/data_structures/field_manager.py), [`AggregateManager`](src/data_structures/aggregate_manager.py), and [`MapManager`](src/data_structures/map_manager.py) classes respectively. New analysis modes can be added by modifying these managers to create additional data objects.

### Merging contributions

This codebase is open to pull requests that add new or improved functionality, subject to the following guidelines:

- PRs should not mix improvements with new functionality.
- PRs that improve existing functionality should pass all acceptance tests without modifying existing test cases, except where correcting errors.
- PRs that add new functionality should minimally modify existing test cases where necessary to ensure all acceptance tests pass. New functionality should also be accompanied by new test cases where necessary.
- All equations and algorithms employed should be suitably referenced. Primary sources are preferred.
- Authors of PRs are required to accept the [contributor licence agreement](CLA.md).

## Config options

### Project options

Options controlling filesystem structure.

| Option                | Description                             |
|-----------------------|-----------------------------------------|
| `ebsd_data_dir`       | Directory for EBSD data files.          |
| `phase_data_dir`      | Directory for phase data files.         |
| `analysis_output_dir` | Directory for generated analysis files. |
| `map_output_dir`      | Directory for generated maps.           |
| `cache_dir`           | Directory for caching reused data.      |
| `phase_database_path` | Path of Pathfinder database file.       |

### Data options

Options specifying additional necessary data not provided in Pathfinder data files.

| Option           | Description                                  |
|------------------|----------------------------------------------|
| `euler_axis_set` | Axis set used for Euler angles in EBSD data. |
| `pixel_size`     | Size of pixels in EBSD data (μm).            |

### Analysis options

Options controlling which analyses to perform, in addition to some common settings.

| Option                          | Description                                                          |
|---------------------------------|----------------------------------------------------------------------|
| `reduce_resolution`             | Perform resolution reduction prior to analysis.                      |
| `compute_dislocation_densities` | Perform GND density analysis.                                        |
| `compute_channelling_fractions` | Perform channelling fraction analysis.                               |
| `compute_orientation_clusters`  | Perform orientation clustering analysis.                             |
| `use_cache`                     | Improves performance by caching reused data.                         |
| `use_cuda`                      | Improves performance by using a CUDA-compatible GPU where available. |
| `random_seed`                   | Seed for random number generation.                                   |

### Map options

Options for generation of map images.

| Option           | Description                                                  |
|------------------|--------------------------------------------------------------|
| `upscale_factor` | Generated maps are multiplicatively upscaled by this factor. |


### Resolution reduction options

Options for the resolution reduction process.

| Option              | Description                                    |
|---------------------|------------------------------------------------|
| `reduction_factor`  | Number of times to apply 2×2 kernel averaging. |
| `scaling_tolerance` | Tolerance for averaging rotation matrices.     |

### Dislocation density options

Options for the dislocation density analysis.

| Option              | Description                                    |
|---------------------|------------------------------------------------|
| `corrective_factor` | Corrective factor for Nye optimisation method. |

### Channelling fraction options

Options for ion channelling fraction analysis.

| Option               | Description                                      |
|----------------------|--------------------------------------------------|
| `beam_atomic_number` | Atomic number of ion beam.                       |
| `beam_energy`        | Energy of ion beam (eV).                         |
| `beam_tilt`          | Tilt of ion beam from z-axis in x=0 plane (deg). |

### Orientation clustering options

Options for orientation clustering analysis.

| Option                 | Description                                                          |
|------------------------|----------------------------------------------------------------------|
| `neighbour_threshold`  | Number of closely oriented neighbours required to form a cluster.    |
| `neighbourhood_radius` | Angular distance for points to be considered closely oriented (deg). |

### Test options

Options specifying location of input data for tests.

| Option                 | Description                                              |
|------------------------|----------------------------------------------------------|
| `ebsd_data_dir`        | Directory for test data files.                           |
| `control_analysis_dir` | Directory for analysis files to compare test results to. |
| `control_map_dir`      | Directory map files to compare test results to.          |
| `config_dir`           | Directory for config files used in tests.                |

## Contributors

- Dr O.J. Whiteside, University of Surrey: primary author, repository owner.
- Dr G. Hobler, Vienna University of Technology: author of code to calculate channelling fractions.

## Related publications

- O.J. Whiteside, “Studying Induced Crystal Structure Changes by Electron Backscatter Diffraction”, PhD thesis, University of Surrey, Guildford, UK, 2024, DOI: [10.15126/thesis.901043](https://doi.org/10.15126/thesis.901043).

## Recommended citation format

- O.J. Whiteside, G. Hobler, _Utilities for processing EBSD data_, online, URI: https://github.com/james-whiteside/ebsd-utils.

## References

1. T.J. Ruggles, D.T. Fullwood, “Estimations of bulk geometrically necessary dislocation density using high resolution EBSD”, _Ultramicroscopy_, vol. 133, pp. 8‐15, 2013, DOI: [10.1016/j.ultramic.2013.04.011](https://doi.org/10.1016/j.ultramic.2013.04.011).
2. G. Hobler, “Critical angles and low‐energy limits to ion channeling in silicon”, _Radiation Effects and Defects in Solids_, vol. 139, pp. 21‐85, 1996, DOI: [10.1080/10420159608212927](https://doi.org/10.1080/10420159608212927).
3. M. Ester, H.P. Kriegel, J. Sander, X. Xu, “A Density‐Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise” in _Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining_, Portland, OR, USA, 1996, DOI: [10.5555/3001460.3001507](https://dl.acm.org/doi/10.5555/3001460.3001507).
4. J. Jiang, T.B. Britton A.J. Wilkinson, “Measurement of geometrically necessary dislocation density with high resolution electron backscatter diffraction: Effects of detector binning and step size”, _Ultramicroscopy_, vol. 125, pp. 1‐9, 2013, DOI: [10.1016/j.ultramic.2012.11.003](https://doi.org/10.1016/j.ultramic.2012.11.003).
