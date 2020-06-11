# python3
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

"""Package metadata for acme.

This is kept in a separate module so that it can be imported from setup.py, at
a time when acme's dependencies may not have been installed yet.
"""

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = '0'
_MINOR_VERSION = '1'
_PATCH_VERSION = '4'

# Example: '0.4.2'
__version__ = '.'.join([_MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION])
