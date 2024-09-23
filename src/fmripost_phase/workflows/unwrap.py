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
"""Workflows for phase unwrapping."""


def init_unwrap_phase_wf(
    mem_gb: float,
    metadata: dict,
    name: str = 'unwrap_phase_wf',
):
    """Build a workflow to rescale and unwrap phase data with MEDIC."""
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_phase.utils.utils import clean_datasinks

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude',
                'phase',
                'bold_mask',
                'skip_vols',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'unwrapped_phase',
            ],
        ),
        name='outputnode',
    )

    unwrap_phase = pe.Node(
        niu.IdentityInterface(
            fields=[
                'unwrapped_phase',
            ],
        ),
        name='unwrap_phase',
    )

    workflow.connect([
        (inputnode, unwrap_phase, [('phase', 'unwrapped_phase')]),
        (unwrap_phase, outputnode, [('unwrapped_phase', 'unwrapped_phase')]),
    ])  # fmt:skip

    return clean_datasinks(workflow)
