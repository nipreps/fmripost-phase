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
"""Interfaces for working with complex-valued data."""

import os

import nibabel as nb
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _CartesianToPolarInputSpec(BaseInterfaceInputSpec):
    real = File(exists=True, desc='Real-valued image')
    imaginary = File(exists=True, desc='Imaginary-valued image')


class _CartesianToPolarOutputSpec(TraitedSpec):
    magnitude = File(exists=True, desc='Magnitude image')
    phase = File(exists=True, desc='Phase image')


class CartesianToPolar(SimpleInterface):
    """Convert Cartesian (real and imaginary) images into polar (magnitude and phase) images."""

    input_spec = _CartesianToPolarInputSpec
    output_spec = _CartesianToPolarOutputSpec

    def _run_interface(self, runtime):
        from fmripost_phase.utils.complex import to_mag, to_phase

        self._results['magnitude'] = os.path.abspath('magnitude.nii.gz')
        self._results['phase'] = os.path.abspath('phase.nii.gz')
        real_img = nb.load(self.inputs.real)
        real_data = real_img.get_fdata()
        imaginary_img = nb.load(self.inputs.imaginary)
        imaginary_data = imaginary_img.get_fdata()

        magnitude_data = to_mag(real_data, imaginary_data)
        phase_data = to_phase(real_data, imaginary_data)
        magnitude_img = nb.Nifti1Image(magnitude_data, real_img.affine, real_img.header)
        phase_img = nb.Nifti1Image(phase_data, real_img.affine, real_img.header)
        magnitude_img.to_filename(self._results['magnitude'])
        phase_img.to_filename(self._results['phase'])
        return runtime


class _PolarToCartesianInputSpec(BaseInterfaceInputSpec):
    magnitude = File(exists=True, mandatory=True, desc='Magnitude image')
    phase = File(exists=True, mandatory=True, desc='Phase image')


class _PolarToCartesianOutputSpec(TraitedSpec):
    real = File(exists=True, desc='Real-valued image')
    imaginary = File(exists=True, desc='Imaginary-valued image')


class PolarToCartesian(SimpleInterface):
    """Convert polar (magnitude and phase) images into Cartesian (real and imaginary) images."""

    input_spec = _PolarToCartesianInputSpec
    output_spec = _PolarToCartesianOutputSpec

    def _run_interface(self, runtime):
        from fmripost_phase.utils.complex import to_imag, to_real

        self._results['real'] = os.path.abspath('real.nii.gz')
        self._results['imaginary'] = os.path.abspath('imaginary.nii.gz')
        magnitude_img = nb.load(self.inputs.magnitude)
        magnitude_data = magnitude_img.get_fdata()
        phase_img = nb.load(self.inputs.phase)
        phase_data = phase_img.get_fdata()

        real_data = to_real(magnitude_data, phase_data)
        imaginary_data = to_imag(magnitude_data, phase_data)
        real_img = nb.Nifti1Image(real_data, magnitude_img.affine, magnitude_img.header)
        imaginary_img = nb.Nifti1Image(imaginary_data, magnitude_img.affine, magnitude_img.header)
        real_img.to_filename(self._results['real'])
        imaginary_img.to_filename(self._results['imaginary'])
        return runtime


class _Scale2RadiansInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Phase image in arbitrary units')
    scale = traits.Enum(
        'pi',
        '2pi',
        desc='Whether to scale between -pi to pi (pi) or 0 to 2*pi (2pi)',
    )


class _Scale2RadiansOutputSpec(TraitedSpec):
    out_file = File(desc='Phase image in radians (-pi to pi or 0 to 2*pi)')


class Scale2Radians(SimpleInterface):
    """Convert a phase map given in a.u. (e.g., 0-4096) to radians (-pi to pi or 0 to 2*pi)."""

    input_spec = _Scale2RadiansInputSpec
    output_spec = _Scale2RadiansOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        import numpy as np

        im = nb.load(self.inputs.in_file)
        data = im.get_fdata(caching='unchanged')
        hdr = im.header.copy()

        # Rescale to [0, 2*pi]
        data = (data - data.min()) * (2 * np.pi / (data.max() - data.min()))
        if self.inputs.scale == 'pi':
            # Rescale to [-pi, pi]
            data = data - np.pi

            # Round to float32 and clip
            data = np.clip(np.float32(data), -np.pi, np.pi)
        else:
            # Round to float32 and clip
            data = np.clip(np.float32(data), 0, 2 * np.pi)

        hdr.set_data_dtype(np.float32)
        hdr.set_xyzt_units('mm')

        self._results['out_file'] = os.path.abspath(os.path.join(runtime.cwd, 'radians.nii.gz'))
        nb.Nifti1Image(data, None, hdr).to_filename(self._results['out_file'])
        return runtime


class _ConcatenateNoiseInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input data file')
    noise_file = File(exists=True, mandatory=True, desc='input noise file')


class _ConcatenateNoiseOutputSpec(TraitedSpec):
    out_file = File(desc='concatenated data and noise files')
    n_noise_volumes = traits.Int(desc='number of noise volumes')


class ConcatenateNoise(SimpleInterface):
    """Concatenate data and noise NIfTI files."""

    input_spec = _ConcatenateNoiseInputSpec
    output_spec = _ConcatenateNoiseOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import concat_imgs

        self._results['n_noise_volumes'] = nb.load(self.inputs.noise_file).shape[3]
        concat_img = concat_imgs([self.inputs.in_file, self.inputs.noise_file])
        self._results['out_file'] = os.path.abspath(os.path.join(runtime.cwd, 'concat.nii.gz'))
        concat_img.to_filename(self._results['out_file'])
        return runtime


class _SplitNoiseInputSpec(BaseInterfaceInputSpec):
    in_file = File(desc='concatenated data and noise files')
    n_noise_volumes = traits.Int(desc='number of noise volumes')


class _SplitNoiseOutputSpec(TraitedSpec):
    out_file = File(exists=True, mandatory=True, desc='output data map')
    noise_file = File(exists=True, mandatory=True, desc='output noise map')


class SplitNoise(SimpleInterface):
    """Split noise volumes out of a concatenated NIfTI file."""

    input_spec = _SplitNoiseInputSpec
    output_spec = _SplitNoiseOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        img = nb.load(self.inputs.in_file)
        data_img = img.slicer[..., : -self.inputs.n_noise_volumes]
        noise_img = img.slicer[..., -self.inputs.n_noise_volumes :]
        self._results['out_file'] = os.path.abspath(os.path.join(runtime.cwd, 'data.nii.gz'))
        self._results['noise_file'] = os.path.abspath(os.path.join(runtime.cwd, 'noise.nii.gz'))
        data_img.to_filename(self._results['out_file'])
        noise_img.to_filename(self._results['noise_file'])
        return runtime
