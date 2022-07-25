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
import pathlib
from collections import defaultdict

import pkg_resources
import setuptools

# https://www.reddit.com/r/Python/comments/3uzl2a/setuppy_requirementstxt_or_a_combination/
# https://stackoverflow.com/questions/49689880/proper-way-to-parse-requirements-file-after-pip-upgrade-to-pip-10-x-x
links = []
requires = []

with pathlib.Path("install-requirements.txt").open() as requirements_txt:
    for item in pkg_resources.parse_requirements(requirements_txt):
        requires.append(str(item))
        if item.url is not None:
            links.append(str(item.req))
requires_extras = defaultdict(lambda: [])
for extra in ["torch-geometric"]:
    with pathlib.Path(f"install-extras-{extra}.txt").open() as requirements_txt:
        for item in pkg_resources.parse_requirements(requirements_txt):
            requires_extras[extra].append(str(item))

setuptools.setup(
    name="t4c22",
    version="0.0.1",
    author="Christian Eichenberger, Moritz Neun",
    description="",
    url="https://github.com/iarai/NeurIPS2022-traffic4cast",
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["t4c22", "t4c22.*"]),
    extras_require=requires_extras,
    include_package_data=True,
    zip_safe=False,
    platforms="any",
    install_requires=requires,
    dependency_links=links,
)
