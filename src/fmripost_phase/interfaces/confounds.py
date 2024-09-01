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
"""Interfaces for miscellaneous confound extraction methods, including HighCor."""

import os
import re

import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class GatherConfoundsInputSpec(BaseInterfaceInputSpec):
    signals = File(exists=True, desc='input signals')
    highcor = File(exists=True, desc='input HighCor')
    cos_basis = File(exists=True, desc='input cosine basis')


class GatherConfoundsOutputSpec(TraitedSpec):
    confounds_file = File(exists=True, desc='output confounds file')
    confounds_list = traits.List(traits.Str, desc='list of headers')


class GatherConfounds(SimpleInterface):
    r"""Combine various sources of confounds in one TSV file.

    Modified from fmriprep.interfaces.confounds.GatherConfounds.

    >>> pd.DataFrame({'a': [0.1]}).to_csv('signals.tsv', index=False, na_rep='n/a')
    >>> pd.DataFrame({'b': [0.2]}).to_csv('highcor.tsv', index=False, na_rep='n/a')

    >>> gather = GatherConfounds()
    >>> gather.inputs.signals = 'signals.tsv'
    >>> gather.inputs.highcor = 'highcor.tsv'
    >>> res = gather.run()
    >>> res.outputs.confounds_list
    ['Global signals', 'HighCor']

    >>> pd.read_csv(res.outputs.confounds_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
         a    b
    0  0.1  0.2

    """

    input_spec = GatherConfoundsInputSpec
    output_spec = GatherConfoundsOutputSpec

    def _run_interface(self, runtime):
        combined_out, confounds_list = _gather_confounds(
            signals=self.inputs.signals,
            highcor=self.inputs.highcor,
            cos_basis=self.inputs.cos_basis,
            newpath=runtime.cwd,
        )
        self._results['confounds_file'] = combined_out
        self._results['confounds_list'] = confounds_list
        return runtime


def _gather_confounds(
    signals=None,
    highcor=None,
    cos_basis=None,
    newpath=None,
):
    """Load confounds from the filenames, concatenate together horizontally and save new file."""

    def less_breakable(a_string):
        """hardens the string to different envs (i.e., case insensitive, no whitespace, '#'"""
        return ''.join(a_string.split()).strip('#')

    # Taken from https://stackoverflow.com/questions/1175208/
    # If we end up using it more than just here, probably worth pulling in a well-tested package
    def camel_to_snake(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _adjust_indices(left_df, right_df):
        # This forces missing values to appear at the beginning of the DataFrame
        # instead of the end
        index_diff = len(left_df.index) - len(right_df.index)
        if index_diff > 0:
            right_df.index = range(index_diff, len(right_df.index) + index_diff)
        elif index_diff < 0:
            left_df.index = range(-index_diff, len(left_df.index) - index_diff)

    all_files = []
    confounds_list = []
    for confound, name in (
        (signals, 'Global signals'),
        (highcor, 'HighCor'),
        (cos_basis, 'Cosine basis'),
    ):
        if confound is not None and isdefined(confound):
            confounds_list.append(name)
            if os.path.exists(confound) and os.stat(confound).st_size > 0:
                all_files.append(confound)

    confounds_data = pd.DataFrame()
    for file_name in all_files:  # assumes they all have headings already
        try:
            new = pd.read_csv(file_name, sep='\t')
        except pd.errors.EmptyDataError:
            # No data, nothing to concat
            continue
        for column_name in new.columns:
            new.rename(
                columns={column_name: camel_to_snake(less_breakable(column_name))}, inplace=True
            )

        _adjust_indices(confounds_data, new)
        confounds_data = pd.concat((confounds_data, new), axis=1)

    if newpath is None:
        newpath = os.getcwd()

    combined_out = os.path.join(newpath, 'confounds.tsv')
    confounds_data.to_csv(combined_out, sep='\t', index=False, na_rep='n/a')

    return combined_out, confounds_list
