# Copyright 2026 Can Deniz Kaya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
In this folder, you will find several tests:
* unit tests that runs atomic tests for processing unit classes.
  These tests can be found in the 'ut' folder.
* integration tests that runs run of processor classes and tests whether the
  run suceeds or fails.
  These tests can be found in the 'it' folder.

We use pytest to run tests. We use pytest's decorator to choose the test level
e.g. either a unit test or an integration test.
A test is follow some rules:
* initialize the input of your method.
* run the method you want to test and save the output.
* create the expected output.
* realize the comparison between the expected output and the output from the
  tested method.
"""
