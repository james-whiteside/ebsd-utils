# -*- coding: utf-8 -*-

from src.data_structures.phase import PhaseMissingError, Phase, BravaisLattice
from src.utilities.config import Config
from src.utilities.filestore import load_phase_database_entry, dump_phase
from src.utilities.logging import Logger


def add_phase(global_id: int, config: Config, logger: Logger) -> int:
    logger.info(f"Adding phase {global_id}.")

    try:
        database_entry = load_phase_database_entry(global_id, config.project.database_path)
        logger.info(f"Database entry found for phase {global_id}:")
        logger.info(f"Name: {database_entry.name}")
        logger.info(f"Lattice type: {database_entry.lattice_type.value}")
        logger.info(f"Lattice constants: {", ".join(f"{constant} nm" for constant in database_entry.lattice_constants_nm)}")
        logger.info(f"Lattice angles: {", ".join(f"{angle} deg" for angle in database_entry.lattice_angles_deg)}")
    except FileNotFoundError:
        logger.warn("Phase database missing. Manual data entry required.")
        database_entry = _input_phase_data(global_id, logger)
    except PhaseMissingError as error:
        logger.warn(f"No database entry found for phase {error.global_id}. Manual data entry required.")
        database_entry = _input_phase_data(global_id, logger)

    supplementary_data = _input_supplementary_data(database_entry.lattice_type, logger)
    phase = Phase.from_parts(database_entry, supplementary_data)
    dump_phase(phase, config.project.phase_dir)
    logger.info("Phase added.")
    return global_id


def _input_phase_data(global_id: int, logger: Logger) -> Phase.DatabaseEntry:
    name = logger.input("Enter phase name: ")
    lattice_type = BravaisLattice[logger.input("Enter Bravais lattice Pearson symbol: ").upper()]
    a = float(logger.input("Enter first lattice constant (nm): "))
    b = float(logger.input("Enter second lattice constant (nm): "))
    c = float(logger.input("Enter third lattice constant (nm): "))
    alpha = float(logger.input("Enter first lattice angle (deg): "))
    beta = float(logger.input("Enter second lattice angle (deg): "))
    gamma = float(logger.input("Enter third lattice angle (deg): "))

    return Phase.DatabaseEntry(
        global_id=global_id,
        name=name,
        lattice_type=lattice_type,
        lattice_constants_nm=(a, b, c),
        lattice_angles_deg=(alpha, beta, gamma),
    )


def _input_supplementary_data(lattice_type: BravaisLattice, logger: Logger) -> Phase.SupplementaryData:
    atomic_number = float(logger.input("Enter average atomic number: "))
    atomic_weight = float(logger.input("Enter average atomic weight: "))
    density_cgs = float(logger.input("Enter density (g/cm³): "))
    vibration_amplitude_nm = float(logger.input("Enter thermal vibration amplitude (nm): "))
    diamond_structure = lattice_type is BravaisLattice.CF and logger.input("Does crystal have diamond structure? (Y/N): ").lower() == "y"

    return Phase.SupplementaryData(
        atomic_number=atomic_number,
        atomic_weight=atomic_weight,
        density_cgs=density_cgs,
        vibration_amplitude_nm=vibration_amplitude_nm,
        diamond_structure=diamond_structure,
    )
