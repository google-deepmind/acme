# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphinx configuration.
"""

project = 'Acme'
author = 'DeepMind Technologies Limited'
copyright = '2018, DeepMind Technologies Limited'  # pylint: disable=redefined-builtin
version = ''
release = ''
master_doc = 'index'

extensions = [
    'myst_parser'
]

html_theme = 'sphinx_rtd_theme'
html_logo = 'imgs/acme.png'
html_theme_options = {
    'logo_only': True,
}
html_css_files = [
    'custom.css',
]

templates_path = []
html_static_path = ['_static']
exclude_patterns = ['_build', 'requirements.txt']

