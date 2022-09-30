#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Iterable
from typing import Set
from typing import Tuple

import numpy as np
from geographiclib.geodesic import Geodesic
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon


T4C_BBOXES = {
    "antwerp": {"bounds": [5100100, 5143700, 415300, 464800], "rotate": True},
    "bangkok": {"bounds": [1355400, 1404900, 10030800, 10074400]},
    "barcelona": {"bounds": [4125300, 4174800, 192500, 236100]},
    "berlin": {"bounds": [5235900, 5285400, 1318900, 1362500]},
    "chicago": {"bounds": [4160100, 4209600, -8794500, -8750900]},
    "istanbul": {"bounds": [4081000, 4130500, 2879400, 2923000]},
    "london": {"bounds": [5120500, 5170000, -36900, 6700]},
    "madrid": {"bounds": [4017700, 4067200, -392700, -349100]},
    "melbourne": {"bounds": [-3810600, -3761100, 14475700, 14519300]},
    "moscow": {"bounds": [5550600, 5594200, 3735800, 3785300], "rotate": True},
    "newyork": {"bounds": [4054400, 4103900, -7415800, -7372200]},
    "vienna": {"bounds": [4795300, 4844800, 1617300, 1660900]},
    "warsaw": {"bounds": [5200100, 5249600, 2081700, 2125300]},
    "zurich": {"bounds": [4708300, 4757800, 834500, 878100]},
}

NUM_ROWS = 495
NUM_COLUMNS = 436


def get_bin_bounds(city):
    return T4C_BBOXES[city]["bounds"]


def is_rotate(city):
    return T4C_BBOXES[city].get("rotate", False)


def get_latlon_bounds(city, verbose=False) -> Tuple[Tuple[float, float, float, float], bool]:
    # Grid cells have steps of size 100. LatLon conversion is floored after the 3rd
    # decimal digit (see degree_to_grid() below). Hence grid to degree needs division
    # through 100 * 1000 = 1e5
    bbox = tuple([c / 1e5 for c in get_bin_bounds(city)])
    if verbose:
        print(f"Bounds for {city}: {bbox}")
    lat_min, lat_max, lon_min, lon_max = bbox
    rotate = is_rotate(city)
    assert np.isclose(lat_max - lat_min, (NUM_ROWS if not rotate else NUM_COLUMNS) * 0.001), lat_max - lat_min
    assert np.isclose(lon_max - lon_min, (NUM_COLUMNS if not rotate else NUM_ROWS) * 0.001), lon_max - lon_min
    return bbox, rotate


# Helpers for the T4C Headings


def get_bearing(lat1, lon1, lat2, lon2):
    return Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)["azi1"] % 360


def angle_to_heading(angle):
    #   quadrant   clockwise  ->  data
    #     NW NE       3 0         1 0
    #     SW SE       2 1         3 2
    clockwise_heading = int(angle // 90) % 4
    return [0, 2, 3, 1][clockwise_heading]


def invert_heading(heading):
    return [3, 2, 1, 0][heading]


# Helpers for working with T4C Movie Grids


def degree_to_grid(d, grid_min):
    # Simple flooring of the degree coordinate value using the first 3 digits after the
    # decimal point as denominators. Return value is relative to the grid minimum.
    # The rounding before casting to integer is necessary to avoid an additional float
    # caused by actual floating point values sometimes being slightly below the expected
    # value after a division.
    c = int(np.round((np.floor(d * 1000) / 1000 - grid_min) * 1000))
    return c


def row_col_for_lon_lat(lon, lat, lon_min, lat_min, rotate):
    # Returns the corresponding grid cell for a given LatLon value relative to the grid origin.
    if not rotate:
        row_degree = lat
        row_degree_min = lat_min
        col_degree = lon
        col_degree_min = lon_min
    else:
        row_degree = lon
        row_degree_min = lon_min
        col_degree = lat
        col_degree_min = lat_min

    row = NUM_ROWS - 1 - degree_to_grid(row_degree, row_degree_min)

    column = degree_to_grid(col_degree, col_degree_min)
    return row, column


def create_polygon_from_row_col(row, col, lon_min, lat_min, rotate) -> Polygon:
    assert 0 <= row < NUM_ROWS, row
    assert 0 <= col < NUM_COLUMNS, col
    if not rotate:
        cell_lat_min = lat_min + (((NUM_ROWS - 1 - row) * 100) / 100000)
        cell_lon_min = lon_min + ((col * 100) / 100000)
    else:
        cell_lat_min = lat_min + (((col) * 100) / 100000)
        cell_lon_min = lon_min + (((NUM_ROWS - 1 - row) * 100) / 100000)
    cell_lat_max = cell_lat_min + (100 / 100000)
    cell_lon_max = cell_lon_min + (100 / 100000)
    top_left = Point(cell_lon_min, cell_lat_max)
    bottom_left = Point(cell_lon_min, cell_lat_min)
    bottom_right = Point(cell_lon_max, cell_lat_min)
    top_right = Point(cell_lon_max, cell_lat_max)
    return Polygon([top_left, bottom_left, bottom_right, top_right, top_left])


def _add_candidate_cells(grid_cells: Set[Tuple[int, int, int]], lon, lat, lon_min, lat_min, rotate, angle, angle_margin=10):
    heading = angle_to_heading(angle)
    # Cope for angles close to the next heading quadrant (margin in degrees)
    heading_buffer1 = angle_to_heading(angle - angle_margin)
    heading_buffer2 = angle_to_heading(angle + angle_margin)
    row, column = row_col_for_lon_lat(lon, lat, lon_min, lat_min, rotate)
    if 0 <= row < NUM_ROWS and 0 <= column < NUM_COLUMNS:
        grid_cells.add((row, column, heading))
        grid_cells.add((row, column, heading_buffer1))
        grid_cells.add((row, column, heading_buffer2))


def add_candidate_cells(grid_cells, lon, lat, lon_min, lat_min, rotate, angle, position_margin=0.00005):
    _add_candidate_cells(grid_cells, lon, lat, lon_min, lat_min, rotate, angle)
    # Cope for close by cells (default margin 0.00005 corresponds to ~5m)
    _add_candidate_cells(grid_cells, lon + position_margin, lat + position_margin, lon_min, lat_min, rotate, angle)
    _add_candidate_cells(grid_cells, lon - position_margin, lat - position_margin, lon_min, lat_min, rotate, angle)


GridCellTuple = Tuple[int, int, int]


def compute_grid_cell_overlaps(line: LineString, grid_cells: Iterable[GridCellTuple], lon_min: float, lat_min: float, rotate: bool):
    # Compute the overlap with the line
    result_list = []
    for row, column, heading in grid_cells:
        cell_polygon = create_polygon_from_row_col(row, column, lon_min, lat_min, rotate)
        overlap_line = line.intersection(cell_polygon)
        if line.length == 0:
            overlap = 0.0
        else:
            overlap = overlap_line.length / line.length
        result_list.append((row, column, heading, overlap))
    return result_list


def get_intersecting_grid_cells(
    line: LineString, lon_min: float, lat_min: float, reverse: bool = False, rotate: bool = False, interpolate: bool = True, step_size: float = 0.0003
):
    if interpolate:
        # Interpolation adds additional points to make sure we catch all cells (default 0.0003~=30m).
        lc = [line.interpolate(d) for d in np.arange(0, line.length + step_size, step_size)]
        line = LineString(lc)
    # Compute the intersecting cells in the corresponding geometry direction.
    grid_cells = set()
    for (lon_from, lat_from), (lon_to, lat_to) in zip(line.coords, line.coords[1:]):
        alpha = get_bearing(lat_from, lon_from, lat_to, lon_to)
        if reverse:
            alpha = (alpha + 180) % 360
            add_candidate_cells(grid_cells, lon_from, lat_from, lon_min, lat_min, rotate, alpha)
            add_candidate_cells(grid_cells, lon_to, lat_to, lon_min, lat_min, rotate, alpha)
        else:
            add_candidate_cells(grid_cells, lon_from, lat_from, lon_min, lat_min, rotate, alpha)
            add_candidate_cells(grid_cells, lon_to, lat_to, lon_min, lat_min, rotate, alpha)
    return compute_grid_cell_overlaps(line, grid_cells, lon_min, lat_min, rotate)
