# -*- coding: utf-8 -*-

from collections.abc import Iterator
from os import makedirs
from typing import Any
from json import dump as dump_json, load as load_json
from xml.etree import ElementTree
from src.data_structures.field import FieldNullError
from src.data_structures.map import Map
from src.data_structures.phase import Phase, BravaisLattice, PhaseMissingError
from src.data_structures.analysis import Analysis
from src.utilities.config import Config
from src.utilities.geometry import Axis


def load_from_data(data_path: str, config: Config, data_ref: str = None) -> Analysis:
    if data_ref is None:
        data_ref = data_path.split("/")[-1].split(".")[0]

    with open(data_path, "r", encoding="utf-8") as file:
        phases: dict[int, Phase] = dict()
        file.readline()
        local_unindexed_id = None

        while True:
            line = file.readline().rstrip("\n").split(",")

            if line == ["Map Size:"]:
                break

            local_id = int(line[0])
            global_id = int(line[2])

            if global_id == Phase.GLOBAL_UNINDEXED_ID:
                local_unindexed_id = local_id
                continue

            try:
                phases[local_id] = load_phase(global_id, config.project.phase_dir)
            except FileNotFoundError:
                raise PhaseMissingError(global_id)

        width = int(file.readline().rstrip("\n").split(",")[1])
        height = int(file.readline().rstrip("\n").split(",")[1])
        phase_id_values: list[list[int | None]] = list()
        euler_angle_values: list[list[tuple[float, float, float] | None]] = list()
        index_quality_values: list[list[float]] = list()
        pattern_quality_values: list[list[float]] = list()
        file.readline()
        file.readline()

        for y in range(height):
            phase_id_values.append(list())
            euler_angle_values.append(list())
            index_quality_values.append(list())
            pattern_quality_values.append(list())

            for x in range(width):
                line = file.readline().rstrip("\n").split(",")
                local_phase_id = int(line[2])

                if local_phase_id == local_unindexed_id:
                    phase_id_values[y].append(None)
                    euler_angle_values[y].append(None)
                else:
                    phase_id_values[y].append(local_phase_id)
                    euler_angle_values[y].append((float(line[3]), float(line[4]), float(line[5])))

                index_quality_values[y].append(float(line[6]))
                pattern_quality_values[y].append(float(line[7]))

    return Analysis(
        data_ref=data_ref,
        width=width,
        height=height,
        phases=phases,
        phase_id_values=phase_id_values,
        euler_angle_values=euler_angle_values,
        pattern_quality_values=pattern_quality_values,
        index_quality_values=index_quality_values,
        config=config,
        local_unindexed_id=local_unindexed_id,
    )


def dump_analysis(analysis: Analysis, dir: str) -> None:
    makedirs(dir, exist_ok=True)
    path = f"{dir}/{analysis.params.analysis_ref}.csv"

    with open(path, "w", encoding="utf-8") as file:
        for row in _analysis_rows(analysis):
            file.write(f"{row}\n")


def _analysis_rows(analysis: Analysis) -> Iterator[str]:
    for row in _analysis_metadata_rows(analysis):
        yield row

    if analysis.config.analysis.compute_clustering:
        for row in _analysis_cluster_aggregate_rows(analysis):
            yield row

    for row in _analysis_data_rows(analysis):
        yield row


def _analysis_metadata_rows(analysis: Analysis) -> Iterator[str]:
    yield "Phases:"

    if analysis.field.phase.has_null_value:
        yield f"{analysis.local_unindexed_id},Not indexed,{Phase.GLOBAL_UNINDEXED_ID}"

    for local_id, phase in analysis.params.phases.items():
        yield f"{local_id},{phase.name},{phase.global_id}"

    yield f"Map size:"
    yield f"X,{analysis.params.width}"
    yield f"Y,{analysis.params.height}"
    yield f"Map scale:"
    yield f"Pixel size (μm),{analysis.params.pixel_size_microns}"

    if analysis.config.analysis.compute_channelling:
        yield f"Channelling:"
        yield f"Atomic number,{analysis.config.channelling.beam_atomic_number}"
        yield f"Energy (eV),{analysis.config.channelling.beam_energy}"
        yield f"Tilt (deg),{analysis.config.channelling.beam_tilt_deg}"

    if analysis.config.analysis.compute_clustering:
        yield f"Clustering:"
        yield f"Core point threshold,{analysis.config.clustering.core_point_threshold}"
        yield f"Point neighbourhood radius (deg),{analysis.config.clustering.neighbourhood_radius_deg}"
        yield f"Cluster count,{analysis.cluster_count}"


def _analysis_cluster_aggregate_rows(analysis: Analysis) -> Iterator[str]:
    yield "Cluster aggregates:"
    columns: list[str] = list()
    columns += ["Cluster ID"]
    columns += ["Cluster Size"]
    columns += ["Phase"]
    columns += ["Euler1", "Euler2", "Euler3"]
    columns += ["Index Quality"]
    columns += ["Pattern Quality"]

    if analysis.config.analysis.compute_channelling:
        columns += ["Beam-IPF x-coordinate", "Beam-IPF y-coordinate"]

    columns += ["Kernel Average Misorientation"]

    if analysis.config.analysis.compute_dislocation:
        columns += ["GND Density"]

    if analysis.config.analysis.compute_channelling:
        columns += ["Channelling Fraction"]

    yield ",".join(columns)

    for id in analysis.cluster_aggregate.group_ids:
        columns: list[str] = list()
        columns += [str(id)]
        columns += analysis.cluster_aggregate.count.serialize_value_for(id)
        columns += analysis.cluster_aggregate._phase_id.serialize_value_for(id)
        columns += analysis.cluster_aggregate.euler_angles_deg.serialize_value_for(id, sig_figs=6)
        columns += analysis.cluster_aggregate.index_quality.serialize_value_for(id, sig_figs=6)
        columns += analysis.cluster_aggregate.pattern_quality.serialize_value_for(id, sig_figs=6)

        if analysis.config.analysis.compute_channelling:
            columns += analysis.cluster_aggregate.ipf_coordinates(analysis.config.channelling.beam_axis).serialize_value_for(id, sig_figs=6)

        columns += analysis.cluster_aggregate.average_misorientation_deg.serialize_value_for(id, sig_figs=6)

        if analysis.config.analysis.compute_dislocation:
            columns += analysis.cluster_aggregate.gnd_density_log.serialize_value_for(id, sig_figs=6)

        if analysis.config.analysis.compute_channelling:
            columns += analysis.cluster_aggregate.channelling_fraction.serialize_value_for(id, sig_figs=6)

        yield ",".join(columns)


def _analysis_data_rows(analysis: Analysis) -> Iterator[str]:
    yield "Data:"
    columns: list[str] = list()
    columns += ["X", "Y"]
    columns += ["Phase"]
    columns += ["Euler1", "Euler2", "Euler3"]
    columns += ["Index Quality"]
    columns += ["Pattern Quality"]

    if analysis.config.analysis.compute_channelling:
        columns += ["Beam-IPF x-coordinate", "Beam-IPF y-coordinate"]

    columns += ["Kernel Average Misorientation"]

    if analysis.config.analysis.compute_dislocation:
        columns += ["GND Density"]

    if analysis.config.analysis.compute_channelling:
        columns += ["Channelling Fraction"]

    if analysis.config.analysis.compute_clustering:
        columns += ["Point Category", "Point Cluster"]

    yield ",".join(columns)

    for y in range(analysis.params.height):
        for x in range(analysis.params.width):
            columns = list()
            columns += [str(x), str(y)]
            columns += analysis.field._phase_id.serialize_value_at(x, y, null_serialization=str(analysis.local_unindexed_id))
            columns += analysis.field.euler_angles_deg.serialize_value_at(x, y, sig_figs=6)
            columns += analysis.field.index_quality.serialize_value_at(x, y, sig_figs=6)
            columns += analysis.field.pattern_quality.serialize_value_at(x, y, sig_figs=6)

            if analysis.config.analysis.compute_channelling:
                columns += analysis.field.ipf_coordinates(analysis.config.channelling.beam_axis).serialize_value_at(x, y, sig_figs=6)

            columns += analysis.field.average_misorientation_deg.serialize_value_at(x, y, sig_figs=6)

            if analysis.config.analysis.compute_dislocation:
                columns += analysis.field.gnd_density_log.serialize_value_at(x, y, sig_figs=6)

            if analysis.config.analysis.compute_channelling:
                columns += analysis.field.channelling_fraction.serialize_value_at(x, y, sig_figs=6)

            if analysis.config.analysis.compute_clustering:
                try:
                    columns += [analysis.field.clustering_category.get_value_at(x, y).code]
                except FieldNullError:
                    columns += [""]

                columns += analysis.field.orientation_cluster_id.serialize_value_at(x, y)

            yield ",".join(columns)


def dump_maps(analysis: Analysis, dir: str):
    dir = f"{dir}/{analysis.params.analysis_ref}"
    makedirs(dir, exist_ok=True)

    for name, map in _analysis_maps(analysis):
        path = f"{dir}/{name}.png"
        map.image.save(path)


def _analysis_maps(analysis: Analysis) -> Iterator[str, Map]:
    yield "phase", analysis.map.phase
    yield "euler_angle", analysis.map.euler_angle
    yield "pattern_quality", analysis.map.pattern_quality
    yield "index_quality", analysis.map.index_quality
    yield "orientation_x", analysis.map.orientation(Axis.X)
    yield "orientation_y", analysis.map.orientation(Axis.Y)
    yield "orientation_z", analysis.map.orientation(Axis.Z)
    yield "average_misorientation", analysis.map.average_misorientation

    if analysis.config.analysis.compute_dislocation:
        yield "gnd_density", analysis.map.gnd_density

    if analysis.config.analysis.compute_channelling:
        yield "orientation_beam", analysis.map.orientation(analysis.config.channelling.beam_axis)
        yield "channelling_fraction", analysis.map.channelling_fraction

    if analysis.config.analysis.compute_clustering:
        yield "orientation_cluster", analysis.map.orientation_cluster


def load_phase(global_id: int, dir: str) -> Phase:
    file_path = f"{dir}/{global_id}.json"

    with open(file_path, "r") as file:
        json_rep: dict[str, Any] = load_json(file)

        kwargs = {
            "global_id": json_rep["global_id"],
            "name": json_rep["name"],
            "atomic_number": json_rep["atomic_number"],
            "atomic_weight": json_rep["atomic_weight"],
            "density_cgs": json_rep["density_cgs"],
            "vibration_amplitude_nm": json_rep["vibration_amplitude_nm"],
            "lattice_type": BravaisLattice(json_rep["lattice_type"]),
            "lattice_constants_nm": tuple(json_rep["lattice_constants_nm"]),
            "lattice_angles_deg": tuple(json_rep["lattice_angles_deg"]),
            "diamond_structure": json_rep["diamond_structure"],
        }

        return Phase(**kwargs)


def dump_phase(phase: Phase, dir: str) -> None:
    makedirs(dir, exist_ok=True)

    json_rep = {
        "global_id": phase.global_id,
        "name": phase.name,
        "atomic_number": phase.atomic_number,
        "atomic_weight": phase.atomic_weight,
        "density_cgs": phase.density_cgs,
        "vibration_amplitude_nm": phase.vibration_amplitude_nm,
        "lattice_type": phase.lattice_type.value,
        "lattice_constants_nm": list(phase.lattice_constants_nm),
        "lattice_angles_deg": list(phase.lattice_angles_deg),
        "diamond_structure": phase.diamond_structure,
    }

    with open(f"{dir}/{phase.global_id}.json", "w") as file:
        dump_json(json_rep, file)


def load_phase_database_entry(global_id: int, path: str) -> Phase.DatabaseEntry:
    database = ElementTree.parse(path).getroot()

    for phase_info in database.iter("CrystalPhaseInfo"):
        if int(phase_info.find("CrystalID").text) == global_id:
            name = phase_info.find("ElementName").text
            lattice_type = BravaisLattice.from_code(int(phase_info.find("BravaisLatticeID").text))
            a = float(phase_info.find("Cell_A").text)
            b = float(phase_info.find("Cell_B").text)
            c = float(phase_info.find("Cell_C").text)
            alpha = float(phase_info.find("Cell_Alpha").text)
            beta = float(phase_info.find("Cell_Beta").text)
            gamma = float(phase_info.find("Cell_Gamma").text)

            return Phase.DatabaseEntry(
                global_id=global_id,
                name=name,
                lattice_type=lattice_type,
                lattice_constants_nm=(a, b, c),
                lattice_angles_deg=(alpha, beta, gamma),
            )

    raise PhaseMissingError(global_id)
