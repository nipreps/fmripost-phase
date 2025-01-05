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
"""Interfaces for the warpkit toolbox."""
import os

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class _MEDICInputSpec(CommandLineInputSpec):
    magnitude = traits.List(
        File(exists=True),
        argstr='--magnitude %s',
        mandatory=True,
        minlen=2,
        desc='Magnitude image(s) to verify registration',
    )
    phase = traits.List(
        File(exists=True),
        argstr='--phase %s',
        mandatory=True,
        minlen=2,
        desc='Phase image(s) to verify registration',
    )
    metadata = traits.List(
        File(exists=True),
        argstr='--metadata %s',
        mandatory=True,
        minlen=2,
        desc='Metadata corresponding to the inputs',
    )
    prefix = traits.Str(
        'medic',
        argstr='--out_prefix %s',
        usedefault=True,
        desc='Prefix for output files',
    )
    noise_frames = traits.Int(
        0,
        argstr='--noiseframes %d',
        usedefault=True,
        desc='Number of noise frames to remove',
    )
    n_cpus = traits.Int(
        4,
        argstr='--n_cpus %d',
        usedefault=True,
        desc='Number of CPUs to use',
    )
    debug = traits.Bool(
        False,
        argstr='--debug',
        usedefault=True,
        desc='Enable debugging output',
    )
    wrap_limit = traits.Bool(
        False,
        argstr='--wrap_limit',
        usedefault=True,
        desc='Turns off some heuristics for phase unwrapping',
    )


class _MEDICOutputSpec(TraitedSpec):
    native_field_map = File(
        exists=True,
        desc='4D native (distorted) space field map in Hertz',
    )
    displacement_map = File(
        exists=True,
        desc='4D displacement map in millimeters',
    )
    field_map = File(
        exists=True,
        desc='4D undistorted field map in Hertz',
    )


class MEDIC(CommandLine):
    """Run MEDIC."""

    _cmd = 'medic'
    input_spec = _MEDICInputSpec
    output_spec = _MEDICOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = os.getcwd()
        outputs['native_field_map'] = os.path.join(
            out_dir,
            f'{self.inputs.prefix}_fieldmaps_native.nii',
        )
        outputs['displacement_map'] = os.path.join(
            out_dir,
            f'{self.inputs.prefix}_displacementmaps.nii',
        )
        outputs['field_map'] = os.path.join(out_dir, f'{self.inputs.prefix}_fieldmaps.nii')
        return outputs


class _WarpkitUnwrapInputSpec(CommandLineInputSpec):
    magnitude = traits.List(
        File(exists=True),
        argstr='--magnitude %s',
        mandatory=True,
        minlen=2,
        desc='Magnitude image(s) to verify registration',
    )
    phase = traits.List(
        File(exists=True),
        argstr='--phase %s',
        mandatory=True,
        minlen=2,
        desc='Phase image(s) to unwrap',
    )
    metadata = traits.List(
        File(exists=True),
        argstr='--metadata %s',
        mandatory=True,
        minlen=2,
        desc='Metadata corresponding to the inputs',
    )
    prefix = traits.Str(
        'warpkit',
        argstr='--out_prefix %s',
        usedefault=True,
        desc='Prefix for output files',
    )
    noise_frames = traits.Int(
        0,
        argstr='--noiseframes %d',
        usedefault=True,
        desc='Number of noise frames to remove',
    )
    n_cpus = traits.Int(
        4,
        argstr='--n_cpus %d',
        usedefault=True,
        desc='Number of CPUs to use',
    )
    debug = traits.Bool(
        False,
        argstr='--debug',
        usedefault=True,
        desc='Enable debugging output',
    )
    wrap_limit = traits.Bool(
        False,
        argstr='--wrap_limit',
        usedefault=True,
        desc='Turns off some heuristics for phase unwrapping',
    )


class _WarpkitUnwrapOutputSpec(TraitedSpec):
    unwrapped = traits.List(
        File(exists=True),
        desc='Unwrapped phase data',
    )


class WarpkitUnwrap(CommandLine):
    """Unwrap multi-echo phase data with warpkit."""

    _cmd = 'warpkit_unwrap'
    input_spec = _WarpkitUnwrapInputSpec
    output_spec = _WarpkitUnwrapOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = os.getcwd()
        out_prefix = os.path.join(out_dir, self.inputs.prefix)
        outputs['unwrapped'] = [
            f'{out_prefix}_echo-{i_echo + 1}_phase.nii.gz' for i_echo in
            range(len(self.inputs.phase))
        ]
        return outputs


class _ROMEOUnwrapInputSpec(CommandLineInputSpec):
    phase = File(
        exists=True,
        argstr='--phase %s',
        mandatory=True,
        desc='Phase image',
    )
    magnitude = File(
        exists=True,
        argstr='--magnitude %s',
        mandatory=True,
        desc='Magnitude image',
    )
    mask = traits.Enum(
        'nomask',
        'robustmask',
        argstr='--mask %s',
        usedefault=True,
    )
    echo_times = traits.Float(
        argstr='--echo-times epi %s',
        desc=(
            'Echo time(s) for input data, in milliseconds(?). '
            'Can use "epi <echo time>" for single-echo data.'
        ),
    )
    no_scale = traits.Bool(
        argstr='--no-rescale',
        desc='Deactivate rescaling of input phase to radians',
    )


class _ROMEOUnwrapOutputSpec(TraitedSpec):
    unwrapped = File(
        exists=True,
        desc='Unwrapped phase data',
    )


class ROMEOUnwrap(CommandLine):
    """Unwrap single-echo phase data with ROMEO.

    From the ROMEO documentation, for a single-echo fMRI file,
    we should use:

    >>> romeo -p ph.nii -m mag.nii -k nomask -t epi -o outputdir
    """

    _cmd = 'romeo'
    input_spec = _ROMEOUnwrapInputSpec
    output_spec = _ROMEOUnwrapOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir = os.getcwd()
        outputs['unwrapped'] = os.path.join(out_dir, 'unwrapped.nii')
        return outputs
