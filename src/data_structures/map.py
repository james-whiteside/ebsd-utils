# -*- coding: utf-8 -*-

from PIL.Image import Image
from src.data_structures.field import Field, FieldType, FieldLike, MapField, FieldNullError, FieldTypeError
from src.utilities.utils import colour_wheel


class Map[VALUE_TYPE]:
    def __init__(
        self,
        value_field: FieldLike[VALUE_TYPE],
        coordinates_field: FieldLike[tuple] = None,
        max_value: VALUE_TYPE = None,
        min_value: VALUE_TYPE = None,
        upscale_factor: int = 1,
    ):
        self.upscale_factor = upscale_factor

        if not value_field.field_type.mappable:
            raise FieldTypeError.lacks_property(value_field.field_type, "mappable")
        else:
            self._values = value_field

        if coordinates_field is None:
            coordinate_values = list()

            for y in range(self._values.height):
                coordinate_values.append(list())

                for x in range(self._values.width):
                    coordinate_values[y].append((x, y))

            self._coordinates = Field.from_array(self._values.width, self._values.height, FieldType.VECTOR_2D, coordinate_values)
        elif coordinates_field.field_type is not FieldType.VECTOR_2D:
            raise FieldTypeError.wrong_type(coordinates_field.field_type, FieldType.VECTOR_2D)
        else:
            self._coordinates = coordinates_field

        if max_value is None:
            self._max_value = self._values.max_value
        else:
            self._max_value = max_value

        if min_value is None:
            self._min_value = self._values.min_value
        else:
            self._min_value = min_value

        self._width = self._values.width
        self._height = self._values.height

    @property
    def field(self) -> MapField:
        match self._values.field_type:
            case FieldType.DISCRETE:
                default_value = (0.0, 0.0, 0.0)
                field = MapField(self._width, self._height, default_value, self.upscale_factor)

                for y in range(self._height):
                    for x in range(self._width):
                        try:
                            value = colour_wheel(self._values.get_value_at(x, y), self._max_value)
                        except FieldNullError:
                            continue

                        field.set_value_at(x, y, value)

            case FieldType.SCALAR:
                default_value = (1.0, 0.0, 0.0)
                field = MapField(self._width, self._height, default_value, self.upscale_factor)

                for y in range(self._height):
                    for x in range(self._width):
                        try:
                            rgb_intensity = (self._values.get_value_at(x, y) - self._min_value) / (self._max_value - self._min_value)
                        except FieldNullError:
                            continue

                        value = (rgb_intensity, rgb_intensity, rgb_intensity)
                        field.set_value_at(x, y, value)

            case FieldType.VECTOR_3D:
                default_value = (0.0, 0.0, 0.0)
                field = MapField(self._width, self._height, default_value, self.upscale_factor)

                for y in range(self._height):
                    for x in range(self._width):
                        try:
                            r_intensity = (self._values.get_value_at(x, y)[0] - self._min_value[0]) / (self._max_value[0] - self._min_value[0])
                            g_intensity = (self._values.get_value_at(x, y)[1] - self._min_value[1]) / (self._max_value[1] - self._min_value[1])
                            b_intensity = (self._values.get_value_at(x, y)[2] - self._min_value[2]) / (self._max_value[2] - self._min_value[2])
                        except FieldNullError:
                            continue

                        value = (r_intensity, g_intensity, b_intensity)
                        field.set_value_at(x, y, value)

            case _:
                raise FieldTypeError.lacks_property(self._values.field_type, "mappable")

        return field

    @property
    def image(self) -> Image:
        return self.field.to_image()
