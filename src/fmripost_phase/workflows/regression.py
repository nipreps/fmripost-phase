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
"""Workflows for phase regression."""

from fmripost_phase import config


def init_phase_regression_wf(name_source, metadata):
    """Build a workflow that denoises a BOLD series using phase regression.

    Parameters
    ----------
    name_source : str
        Path to the raw magnitude BOLD series file.
    metadata : dict
        Metadata dictionary from BIDS JSON file.

    Inputs
    ------
    magnitude : str
        Path to the magnitude BOLD series file after minimal preprocessing, in boldref space.
    phase : str
        Path to the unwrapped phase BOLD series file, in radians.
    bold_mask : str
        Path to the brain mask of the magnitude BOLD series in boldref space.

    Outputs
    -------
    denoised_magnitude : str
        Path to the denoised magnitude BOLD series file.
    phase : str
        Path to the fitted phase regression confound file.

    Notes
    -----
    1.  Remove non-steady-state volumes from phase data (identified from magnitude processing).
    2.  Compute optimal order of Legendre polynomials.
    3.  ~~Convert phase from arbitrary units to radians.~~
    4.  Combine magnitude and phase into complex-valued data.
    5.  Split complex-valued data into real and imaginary components.
    6.  Apply motion correction transform from magnitude processing to real and imaginary files.
    7.  Combine motion-corrected real and imaginary files into complex-valued data.
    8.  Split motion-corrected complex-valued file into magnitude and phase components.
    9.  ~~Temporally unwrap the motion-corrected phase data.~~
    10. Detrend the unwrapped phase data using the Legendre polynomial order determined previously.
    11. Apply brain mask from magnitude processing to phase data.
    12. ODR phase regression.

        1.  Compute the power spectrum map of the phase data
        2.  Find the time point index of the maximum frequency peak starting from the 93th time
            point.
        3.  Compute the value that happens the most in the previous map to find the time point
            index of the respiration peak.
        4.  Multiply this index by 1.1 to consider the subpeaks around the maximum peak.
        5.  Convert it to frequencies (Hz) by multiplying it with the frequency value of the first
            time point.
        6.  Run the ODR regression.
    13. ~~Run standard GLM on phase-regressed magnitude data.~~

        -   I'm not sure why the GLM is done separately from the phase regression.
            Maybe because the phase regression requires ODR?
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_phase.interfaces.complex import CartesianToPolar, PolarToCartesian
    from fmripost_phase.interfaces.regression import ODRFit
    from fmripost_phase.utils.utils import _get_wf_name

    workflow = Workflow(name=_get_wf_name(name_source, 'phase_regression'))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude',
                'phase',
                'bold_mask',
                'space',
                'cohort',
                'res',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'denoised_magnitude',
                'phase',
            ],
        ),
        name='outputnode',
    )

    # Compute optimal order of Legendre polynomials
    determine_leg_order = pe.Node(
        niu.IdentityInterface(fields=['phase']),
        name='determine_leg_order',
    )
    workflow.connect([(inputnode, determine_leg_order, [('phase', 'phase')])])

    # Convert polar data to real and imaginary data
    convert_to_cartesian = pe.Node(
        PolarToCartesian(),
        name='convert_to_cartesian',
    )
    workflow.connect([
        (inputnode, convert_to_cartesian, [
            ('magnitude', 'magnitude'),
            ('phase', 'phase'),
        ]),
    ])  # fmt:skip

    # Apply motion correction transform from magnitude processing to real and imaginary files.
    # XXX: Why apply motion correction to imaginary data instead of phase data?
    apply_motion_correction = pe.Node(
        niu.IdentityInterface(fields=['real', 'imaginary']),
        name='apply_motion_correction',
    )
    workflow.connect([
        (convert_to_cartesian, apply_motion_correction, [
            ('real', 'real'),
            ('imaginary', 'imaginary'),
        ]),
    ])  # fmt:skip

    # Combine motion-corrected real and imaginary files into polar data.
    convert_to_polar = pe.Node(
        CartesianToPolar(),
        name='convert_to_polar',
    )
    workflow.connect([
        (apply_motion_correction, convert_to_polar, [
            ('real', 'real'),
            ('imaginary', 'imaginary'),
        ]),
    ])  # fmt:skip

    # Detrend the unwrapped phase data using the Legendre polynomial order determined previously.
    detrend_phase = pe.Node(
        niu.IdentityInterface(fields=['phase', 'order']),
        name='detrend_phase',
    )
    workflow.connect([
        (determine_leg_order, detrend_phase, [('order', 'order')]),
        (convert_to_polar, detrend_phase, [('phase', 'phase')]),
    ])  # fmt:skip

    # Apply brain mask from magnitude processing to phase data.
    apply_mask = pe.Node(
        niu.IdentityInterface(fields=['phase', 'mask_file']),
        name='apply_mask',
    )
    workflow.connect([
        (inputnode, apply_mask, [('bold_mask', 'mask_file')]),
        (detrend_phase, apply_mask, [('phase', 'phase')]),
    ])  # fmt:skip

    # ODR phase regression
    odr_regression = pe.Node(
        ODRFit(TR=metadata['RepetitionTime'], noise_filter=config.workflow.noise_filter),
        name='odr_regression',
    )
    workflow.connect([
        (inputnode, odr_regression, [
            ('magnitude', 'magnitude'),
            ('bold_mask', 'mask'),
        ]),
        (apply_mask, odr_regression, [('phase', 'phase')]),
        (odr_regression, outputnode, [
            ('sim', 'denoised_magnitude'),
            ('estimate', 'phase'),
        ]),
    ])  # fmt:skip

    return workflow
