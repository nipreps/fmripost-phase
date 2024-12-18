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
)


class _MagnitudePhaseToComplexInputSpec(BaseInterfaceInputSpec):
    magnitude_file = File(exists=True, mandatory=True, desc='Magnitude image')
    phase_file = File(exists=True, mandatory=True, desc='Phase image')


class _MagnitudePhaseToComplexOutputSpec(TraitedSpec):
    complex_file = File(exists=True, desc='Complex-valued image')


class MagnitudePhaseToComplex(SimpleInterface):
    """Combine magnitude and phase images into a complex-valued image."""

    input_spec = _MagnitudePhaseToComplexInputSpec
    output_spec = _MagnitudePhaseToComplexOutputSpec

    def _run_interface(self, runtime):
        from fmripost_phase.utils.complex import imgs_to_complex

        self._results['complex_file'] = os.path.abspath('complex.nii.gz')
        magnitude_img = nb.load(self.inputs.magnitude_file)
        phase_img = nb.load(self.inputs.phase_file)
        complex_img = imgs_to_complex(magnitude_img, phase_img)
        complex_img.to_filename(self._results['complex_file'])
        return runtime


class _ComplexToMagnitudePhaseInputSpec(BaseInterfaceInputSpec):
    complex_file = File(exists=True, mandatory=True, desc='Complex-valued image')


class _ComplexToMagnitudePhaseOutputSpec(TraitedSpec):
    magnitude_file = File(exists=True, desc='Magnitude image')
    phase_file = File(exists=True, desc='Phase image')


class ComplexToMagnitudePhase(SimpleInterface):
    """Split a complex-valued image into magnitude and phase images."""

    input_spec = _ComplexToMagnitudePhaseInputSpec
    output_spec = _ComplexToMagnitudePhaseOutputSpec

    def _run_interface(self, runtime):
        from fmripost_phase.utils.complex import split_complex

        self._results['magnitude_file'] = os.path.abspath('magnitude.nii.gz')
        self._results['phase_file'] = os.path.abspath('phase.nii.gz')
        complex_img = nb.load(self.inputs.complex_file)
        magnitude_img, phase_img = split_complex(complex_img)
        magnitude_img.to_filename(self._results['magnitude_file'])
        phase_img.to_filename(self._results['phase_file'])
        return runtime


class _ComplexToRealImaginaryInputSpec(BaseInterfaceInputSpec):
    complex_file = File(exists=True, mandatory=True, desc='Complex-valued image')


class _ComplexToRealImaginaryOutputSpec(TraitedSpec):
    real_file = File(exists=True, desc='Real-valued image')
    imaginary_file = File(exists=True, desc='Imaginary-valued image')


class ComplexToRealImaginary(SimpleInterface):
    """Convert a complex-valued image into real and imaginary images."""

    input_spec = _ComplexToRealImaginaryInputSpec
    output_spec = _ComplexToRealImaginaryOutputSpec

    def _run_interface(self, runtime):
        self._results['real_file'] = os.path.abspath('real.nii.gz')
        self._results['imaginary_file'] = os.path.abspath('imaginary.nii.gz')
        complex_img = nb.load(self.inputs.complex_file)
        complex_data = complex_img.get_fdata()
        real_data = complex_data.real
        imag_data = complex_data.imag
        real_img = nb.Nifti1Image(real_data, complex_img.affine, complex_img.header)
        imaginary_img = nb.Nifti1Image(imag_data, complex_img.affine, complex_img.header)
        real_img.to_filename(self._results['real_file'])
        imaginary_img.to_filename(self._results['imaginary_file'])
        return runtime


class _RealImaginaryToMagnitudePhaseInputSpec(BaseInterfaceInputSpec):
    real_file = File(exists=True, desc='Real-valued image')
    imaginary_file = File(exists=True, desc='Imaginary-valued image')


class _RealImaginaryToMagnitudePhaseOutputSpec(TraitedSpec):
    magnitude_file = File(exists=True, desc='Magnitude image')
    phase_file = File(exists=True, desc='Phase image')


class RealImaginaryToMagnitudePhase(SimpleInterface):
    """Convert a complex-valued image into real and imaginary images."""

    input_spec = _RealImaginaryToMagnitudePhaseInputSpec
    output_spec = _RealImaginaryToMagnitudePhaseOutputSpec

    def _run_interface(self, runtime):
        from fmripost_phase.utils.complex import to_mag, to_phase

        self._results['magnitude_file'] = os.path.abspath('magnitude.nii.gz')
        self._results['phase_file'] = os.path.abspath('phase.nii.gz')
        real_img = nb.load(self.inputs.real_file)
        real_data = real_img.get_fdata()
        imaginary_img = nb.load(self.inputs.imaginary_file)
        imaginary_data = imaginary_img.get_fdata()

        magnitude_data = to_mag(real_data, imaginary_data)
        phase_data = to_phase(real_data, imaginary_data)
        magnitude_img = nb.Nifti1Image(magnitude_data, real_img.affine, real_img.header)
        phase_img = nb.Nifti1Image(phase_data, real_img.affine, real_img.header)
        magnitude_img.to_filename(self._results['magnitude_file'])
        phase_img.to_filename(self._results['phase_file'])
        return runtime


class _MagnitudePhaseToRealImaginaryInputSpec(BaseInterfaceInputSpec):
    magnitude_file = File(exists=True, mandatory=True, desc='Magnitude image')
    phase_file = File(exists=True, mandatory=True, desc='Phase image')


class _MagnitudePhaseToRealImaginaryOutputSpec(TraitedSpec):
    real_file = File(exists=True, desc='Real-valued image')
    imaginary_file = File(exists=True, desc='Imaginary-valued image')


class MagnitudePhaseToRealImaginary(SimpleInterface):
    """Convert magnitude and phase images into real and imaginary images."""

    input_spec = _MagnitudePhaseToRealImaginaryInputSpec
    output_spec = _MagnitudePhaseToRealImaginaryOutputSpec

    def _run_interface(self, runtime):
        from fmripost_phase.utils.complex import to_imag, to_real

        self._results['real_file'] = os.path.abspath('real.nii.gz')
        self._results['imaginary_file'] = os.path.abspath('imaginary.nii.gz')
        magnitude_img = nb.load(self.inputs.magnitude_file)
        magnitude_data = magnitude_img.get_fdata()
        phase_img = nb.load(self.inputs.phase_file)
        phase_data = phase_img.get_fdata()

        real_data = to_real(magnitude_data, phase_data)
        imaginary_data = to_imag(magnitude_data, phase_data)
        real_img = nb.Nifti1Image(real_data, magnitude_img.affine, magnitude_img.header)
        imaginary_img = nb.Nifti1Image(imaginary_data, magnitude_img.affine, magnitude_img.header)
        real_img.to_filename(self._results['real_file'])
        imaginary_img.to_filename(self._results['imaginary_file'])
        return runtime
