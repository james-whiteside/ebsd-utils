# -*- coding: utf-8 -*-

from enum import Enum
from PIL.Image import Image
from field import Field, FieldType, FieldLike, MapField
from utilities import colour_wheel


class MapType(Enum):
    P = "phase"
    EA = "euler angle"
    PQ = "pattern quality"
    IQ = "index quality"
    OX = "x-axis orientation"
    OY = "y-axis orientation"
    OZ = "z-axis orientation"


class Map[VALUE_TYPE]:
    def __init__(
        self,
        map_type: MapType,
        value_field: FieldLike[VALUE_TYPE],
        filter_field: FieldLike[bool] = None,
        coordinates_field: FieldLike[tuple] = None,
        max_value: VALUE_TYPE = None,
        min_value: VALUE_TYPE = None,
    ):
        self.map_type = map_type

        if not value_field.field_type.mappable:
            raise ValueError(f"Value field is not a mappable field type: {value_field.field_type.name}")
        else:
            self._values = value_field

        if filter_field is None:
            filter_values = list()

            for y in range(self._values.height):
                filter_values.append(list())

                for x in range(self._values.width):
                    filter_values[y].append(True)

            self._filter = Field(self._values.width, self._values.height, FieldType.BOOLEAN, values=filter_values)
        elif filter_field.field_type is not FieldType.BOOLEAN:
            raise ValueError(f"Coordinate field must be {FieldType.BOOLEAN.name}, not {filter_field.field_type.name}.")
        else:
            self._filter = filter_field

        if coordinates_field is None:
            coordinate_values = list()

            for y in range(self._values.height):
                coordinate_values.append(list())

                for x in range(self._values.width):
                    coordinate_values[y].append((x, y))

            self._coordinates = Field(self._values.width, self._values.height, FieldType.VECTOR_2D, values=coordinate_values)
        elif coordinates_field.field_type is not FieldType.VECTOR_2D:
            raise ValueError(f"Coordinate field must be {FieldType.VECTOR_2D.name}, not {coordinates_field.field_type.name}.")
        else:
            self._coordinates = coordinates_field

        if max_value is None:
            self._max_value = max(value for value, filter in zip(self._values.values, self._filter.values) if filter)
        else:
            self._max_value = max_value

        if min_value is None:
            self._min_value = min(value for value, filter in zip(self._values.values, self._filter.values) if filter)
        else:
            self._min_value = min_value

        self._width = self._values.width
        self._height = self._values.height

    @property
    def field(self) -> MapField:
        match self._values.field_type:
            case FieldType.DISCRETE:
                field = MapField(self._width, self._height, default_value=(0.0, 0.0, 0.0))

                for y in range(self._height):
                    for x in range(self._width):
                        if self._filter.get_value_at(x, y):
                            value = colour_wheel(self._values.get_value_at(x, y), self._max_value)
                            field.set_value_at(x, y, value)

            case FieldType.SCALAR:
                field = MapField(self._width, self._height, default_value=(1.0, 0.0, 0.0))

                for y in range(self._height):
                    for x in range(self._width):
                        if self._filter.get_value_at(x, y):
                            rgb_intensity = (self._values.get_value_at(x, y) - self._min_value) / (self._max_value - self._min_value)
                            value = (rgb_intensity, rgb_intensity, rgb_intensity)
                            field.set_value_at(x, y, value)

            case FieldType.VECTOR_3D:
                field = MapField(self._width, self._height, default_value=(0.0, 0.0, 0.0))

                for y in range(self._height):
                    for x in range(self._width):
                        if self._filter.get_value_at(x, y):
                            r_intensity = (self._values.get_value_at(x, y)[0] - self._min_value[0]) / (self._max_value[0] - self._min_value[0])
                            g_intensity = (self._values.get_value_at(x, y)[1] - self._min_value[1]) / (self._max_value[1] - self._min_value[1])
                            b_intensity = (self._values.get_value_at(x, y)[2] - self._min_value[2]) / (self._max_value[2] - self._min_value[2])
                            value = (r_intensity, g_intensity, b_intensity)
                            field.set_value_at(x, y, value)

            case _:
                raise NotImplementedError()

        return field

    @property
    def image(self) -> Image:
        return self.field.to_image()
