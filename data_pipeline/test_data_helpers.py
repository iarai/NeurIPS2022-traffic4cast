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
import pytest
from shapely.geometry import LineString

from data_pipeline.data_helpers import angle_to_heading
from data_pipeline.data_helpers import create_polygon_from_row_col
from data_pipeline.data_helpers import degree_to_grid
from data_pipeline.data_helpers import get_bearing
from data_pipeline.data_helpers import get_intersecting_grid_cells
from data_pipeline.data_helpers import invert_heading
from data_pipeline.data_helpers import row_col_for_lon_lat


def test_get_bearing():
    b = get_bearing(50.062, 8.2, 50.063, 8.3)
    assert abs(b - 89.07) < 0.1


def test_angle_to_heading():
    # quadrant  data
    # NW NE     1 0
    # SW SE     3 2
    assert angle_to_heading(5.0) == 0  # NE
    assert angle_to_heading(91.0) == 2  # SE
    assert angle_to_heading(179.0) == 2  # SE
    assert angle_to_heading(181.0) == 3  # SW
    assert angle_to_heading(271.0) == 1  # NW
    assert angle_to_heading(359.0) == 1  # NW
    assert angle_to_heading(361.0) == 0  # NE
    assert angle_to_heading(-3.0) == 1  # NW


def test_invert_heading():
    assert invert_heading(0) == 3
    assert invert_heading(3) == 0
    assert invert_heading(1) == 2
    assert invert_heading(2) == 1


def test_degree_to_grid():
    assert degree_to_grid(5.12301, 5.0) == 123
    assert degree_to_grid(5.12345, 5.0) == 123
    assert degree_to_grid(5.12399, 5.0) == 123
    assert degree_to_grid(5.12499, 5.0) == 124
    assert degree_to_grid(50.00001, 50.0) == 0
    assert degree_to_grid(50.00101, 50.0) == 1
    assert degree_to_grid(49.99999, 50.0) == -1


@pytest.mark.parametrize(
    "lon_min, lat_min, lon,lat, expected_row, expected_col, rotate",
    [
        [8.0, 50.0, 8.2, 50.0, 494, 200, False],
        [8.0, 50.0, 8.2, 50.02, 474, 200, False],
        [8.0, 50.0, 8.2, 50.494, 0, 200, False],
        [8.0, 50.0, 8.0008, 50.494, 0, 0, False],
        [-8.0, -50.0, -7.8, -50.0, 494, 200, False],
        [-8.0, -50.0, -7.8, -49.98, 474, 200, False],
        [-8.0, -50.0, -7.8, -49.506, 0, 200, False],
        [-8.0, -50.0, -7.9992, -49.506, 0, 0, False],
        [8.0, 50.0, 8.494, 50.0, 0, 0, True],
        [8.0, 50, 8.494, 50.435, 0, 435, True],
        [8.0, 50.0, 8.0, 50.0, 494, 0, True],
        [8.0, 50.0, 8.0, 50.435, 494, 435, True],
    ],
)
def test_row_col_for_lon_lat(lon_min, lat_min, lon, lat, expected_row, expected_col, rotate):
    row, col = row_col_for_lon_lat(lon, lat, lon_min, lat_min, rotate=rotate)
    assert row == expected_row, (row, expected_row)
    assert col == expected_col, (col, expected_col)


# TODO test with negative values!
@pytest.mark.parametrize(
    "lon_min,lat_min,rotate,row,col,expected_coords",
    [
        [8.0, 50.0, False, 433, 432, [(8.432, 50.062), (8.432, 50.061), (8.432999999999999, 50.061), (8.432999999999999, 50.062), (8.432, 50.062)]],
        # rotate=False: lat decreasing in rows, lon increasing in columns
        [8.0, 50.0, False, 0, 0, [(8.0, 50.495), (8.0, 50.494), (8.001, 50.494), (8.001, 50.495), (8.0, 50.495)]],
        [8.0, 50.0, False, 494, 0, [(8.0, 50.001), (8.0, 50.0), (8.001, 50.0), (8.001, 50.001), (8.0, 50.001)]],
        [8.0, 50.0, False, 494, 435, [(8.435, 50.001), (8.435, 50.0), (8.436, 50.0), (8.436, 50.001), (8.435, 50.001)]],
        # rotate=True: lon decreasing in rows, lat increasing in columns
        [8.0, 50.0, True, 0, 0, [(8.494, 50.001), (8.494, 50.0), (8.495, 50.0), (8.495, 50.001), (8.494, 50.001)]],
        [8.0, 50.0, True, 494, 435, [(8.0, 50.436), (8.0, 50.435), (8.001, 50.435), (8.001, 50.436), (8.0, 50.436)]],
    ],
)
def test_create_polygon_from_row_col(lon_min, lat_min, rotate, row, col, expected_coords):
    p = create_polygon_from_row_col(row, col, lon_min, lat_min, rotate=rotate)
    print(p)
    coords = list(p.exterior.coords)
    print(coords)
    for c, ec in zip(coords, expected_coords):
        assert c[0] == ec[0]
        assert c[1] == ec[1]


def assert_intersecting_cells_equal(ic1, ic2):
    assert len(ic1) == len(ic2)
    ic1 = sorted(set(ic1))
    ic2 = sorted(set(ic2))
    assert len(ic1) == len(ic2)
    for (r1, c1, h1, o1), (r2, c2, h2, o2) in zip(ic1, ic2):
        assert r1 == r2
        assert c1 == c2
        assert h1 == h2
        assert abs(o1 - o2) < 0.00001


def test_get_intersecting_grid_cells():
    lat_min = 50.0
    lon_min = 8.0

    line = LineString([(8.011, 50.021), (8.013, 50.022)])
    expected_cells_forward = [(472, 13, 0, 0.0), (473, 12, 0, 0.5), (474, 10, 0, 0.0), (473, 11, 0, 0.5), (473, 10, 0, 0.0)]
    expected_cells_reverse = [(r, c, 3, o) for r, c, _, o in expected_cells_forward]

    cells = get_intersecting_grid_cells(line, lon_min, lat_min)
    assert_intersecting_cells_equal(cells, expected_cells_forward)

    cells = get_intersecting_grid_cells(line, lon_min, lat_min, reverse=True)
    assert_intersecting_cells_equal(cells, expected_cells_reverse)

    line = LineString([(8.011, 50.0211), (8.011, 50.0212)])
    cells = get_intersecting_grid_cells(line, lon_min, lat_min, reverse=False)
    expected_cells_forward = [(473, 11, 0, 1.0), (473, 10, 0, 1.0), (473, 10, 1, 1.0), (473, 11, 1, 1.0)]
    assert_intersecting_cells_equal(cells, expected_cells_forward)
