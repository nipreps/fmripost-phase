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
"""Utilities for working with complex-valued nifti images."""

import nibabel as nb
import numpy as np
from nilearn._utils import check_niimg


def imgs_to_complex(mag, phase):
    """Combine magnitude and phase images into a complex-valued nifti image."""
    mag = check_niimg(mag)
    phase = check_niimg(phase)
    # Convert to radians to be extra safe
    phase = to_radians(phase)
    mag_data = mag.get_fdata()
    phase_data = phase.get_fdata()
    comp_data = to_complex(mag_data, phase_data)
    comp_img = nb.Nifti1Image(comp_data, mag.affine)
    return comp_img


def split_complex(comp_img):
    """Split a complex-valued nifti image into magnitude and phase images."""
    comp_img = check_niimg(comp_img)
    comp_data = comp_img.get_fdata(dtype=comp_img.get_data_dtype())
    real = comp_data.real
    imag = comp_data.imag
    mag = abs(comp_data)
    phase = to_phase(real, imag)
    mag = nb.Nifti1Image(mag, comp_img.affine)
    phase = nb.Nifti1Image(phase, comp_img.affine)
    return mag, phase


def to_complex(mag, phase):
    """Convert magnitude and phase data into complex real+imaginary data.

    Should be equivalent to cmath.rect.
    """
    comp = mag * np.exp(1j * phase)
    return comp


def to_mag(real, imag):
    """Convert real and imaginary data to magnitude data.

    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase
    """
    mag = np.sqrt((real**2) + (imag**2))
    return mag


def to_phase(real, imag):
    """Convert real and imaginary data to phase data.

    Equivalent to cmath.phase.

    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase
    """
    phase = np.arctan2(imag, real)
    return phase


def to_real(mag, phase):
    """Convert magnitude and phase data to real data."""
    comp = to_complex(mag, phase)
    real = comp.real
    return real


def to_imag(mag, phase):
    """Convert magnitude and phase data to imaginary data."""
    comp = to_complex(mag, phase)
    imag = comp.imag
    return imag


def to_radians(in_file):
    """Ensure that phase images are in a usable range for unwrapping [0, 2pi).

    Adapted from
    https://github.com/poldracklab/sdcflows/blob/
    659c2508ecef810c3acadbe808560b44d22801f9/sdcflows/interfaces/fmap.py#L94

    From the FUGUE User guide::

        If you have separate phase volumes that are in integer format then do:

        fslmaths orig_phase0 -mul 3.14159 -div 2048 phase0_rad -odt float
        fslmaths orig_phase1 -mul 3.14159 -div 2048 phase1_rad -odt float

        Note that the value of 2048 needs to be adjusted for each different
        site/scanner/sequence in order to be correct. The final range of the
        phase0_rad image should be approximately 0 to 6.28. If this is not the
        case then this scaling is wrong. If you have separate phase volumes are
        not in integer format, you must still check that the units are in
        radians, and if not scale them appropriately using fslmaths.
    """
    phase_img = nb.load(in_file)
    phase_data = phase_img.get_fdata()
    imax = phase_data.max()
    imin = phase_data.min()
    scaled = (phase_data - imin) / (imax - imin)
    rad_data = 2 * np.pi * scaled
    out_img = nb.Nifti1Image(rad_data, phase_img.affine, phase_img.header)
    out_file = os.path.abspath('phase_radians.nii.gz')
    out_img.to_filename(out_file)
    return out_file
