from field import Field, FieldType, FieldLike
from phase import Phase


class Map[VALUE_TYPE]:
    def __init__(
        self,
        value_field: FieldLike[VALUE_TYPE],
        phase_field: FieldLike[Phase],
        coordinates_field: FieldLike[tuple] = None,
        max_value: VALUE_TYPE = None,
        min_value: VALUE_TYPE = None,
    ):
        if not value_field.field_type.mappable:
            raise ValueError(f"Value field is not a mappable field type: {value_field.field_type.name}")
        else:
            self.values = value_field

        self.phase = phase_field

        if coordinates_field is None:
            coordinate_values = list()

            for y in range(self.values.height):
                coordinate_values.append(list())

                for x in range(self.values.width):
                    coordinate_values[y].append((x, y))

            self.coordinates = Field(self.values.width, self.values.height, FieldType.VECTOR_2D, values=coordinate_values)
        elif coordinates_field.field_type is not FieldType.VECTOR_2D:
            raise ValueError(f"Coordinate field must be {FieldType.VECTOR_2D.name}, not {coordinates_field.field_type.name}.")
        else:
            self.coordinates = coordinates_field

        if max_value is None:
            self.max_value = self.values.max_value
        else:
            self.max_value = max_value

        if min_value is None:
            self.min_value = self.values.min_value
        else:
            self.min_value = min_value
