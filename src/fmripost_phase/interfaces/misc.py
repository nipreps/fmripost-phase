# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Interfaces to perform miscellaneous operations."""

import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _RemoveNSSInputSpec(BaseInterfaceInputSpec):
    in_file = File(desc='input file')
    skip_vols = traits.Int(desc='number of NSS volumes to remove')


class _RemoveNSSOutputSpec(TraitedSpec):
    out_file = File(exists=True, mandatory=True, desc='output file')


class RemoveNSS(SimpleInterface):
    """Remove non-steady-state volumes from a 4D NIfTI."""

    input_spec = _RemoveNSSInputSpec
    output_spec = _RemoveNSSOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        img = nb.load(self.inputs.in_file)
        img = img.slicer[..., self.inputs.skip_vols :]
        self._results['out_file'] = os.path.abspath(os.path.join(runtime.cwd, 'cropped.nii.gz'))
        img.to_filename(self._results['out_file'])
        return runtime
