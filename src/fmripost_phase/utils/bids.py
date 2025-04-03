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
"""Utilities to handle BIDS inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bids.layout import BIDSLayout
from bids.utils import listify
from niworkflows.utils.spaces import SpatialReferences

from fmripost_phase import config
from fmripost_phase.data import load as load_data


def extract_entities(file_list: str | list[str]) -> dict:
    """Return a dictionary of common entities given a list of files.

    Parameters
    ----------
    file_list : str | list[str]
        File path or list of file paths.

    Returns
    -------
    entities : dict
        Dictionary of entities.

    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}

    """
    from collections import defaultdict

    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair for f in listify(file_list) for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def collect_derivatives(
    raw_dataset: Path | BIDSLayout | None,
    derivatives_dataset: Path | BIDSLayout | None,
    entities: dict | None,
    fieldmap_id: str | None,
    spec: dict | None = None,
    patterns: list[str] | None = None,
    allow_multiple: bool = False,
    spaces: SpatialReferences | None = None,
) -> dict:
    """Gather existing derivatives and compose a cache.

    TODO: Ingress 'spaces' and search for BOLD+mask in the spaces *or* xfms.

    Parameters
    ----------
    raw_dataset : Path | BIDSLayout | None
        Path to the raw dataset or a BIDSLayout instance.
    derivatives_dataset : Path | BIDSLayout
        Path to the derivatives dataset or a BIDSLayout instance.
    entities : dict
        Dictionary of entities to use for filtering.
    fieldmap_id : str | None
        Fieldmap ID to use for filtering.
    spec : dict | None
        Specification dictionary.
    patterns : list[str] | None
        List of patterns to use for filtering.
    allow_multiple : bool
        Allow multiple files to be returned for a given query.
    spaces : SpatialReferences | None
        Spatial references to select for.

    Returns
    -------
    derivs_cache : dict
        Dictionary with keys corresponding to the derivatives and values
        corresponding to the file paths.
    """
    if not entities:
        entities = {}

    print(entities)

    bids_filters = config.execution.bids_filters or {}
    bold_filters = bids_filters.get('bold', {})
    anat_filters = bids_filters.get('anat', {})

    if spec is None or patterns is None:
        _spec = json.loads(load_data.readable('io_spec.json').read_text())

        if spec is None:
            spec = _spec['queries']

        if patterns is None:
            patterns = _spec['patterns']

    # Search for derivatives data
    derivs_cache = defaultdict(list, {})
    if derivatives_dataset is not None:
        layout = derivatives_dataset
        if isinstance(layout, Path):
            layout = BIDSLayout(
                layout,
                config=['bids', 'derivatives'],
                validate=False,
            )

        for k, q in spec['derivatives'].items():
            # Combine entities with query. Query values override file entities.
            if k.startswith('bold'):
                query = {**bold_filters, **entities.get('bold', {}), **q}
            elif k.startswith('anat'):
                query = {**anat_filters, **entities.get('anat', {}), **q}

            item = layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

        for k, q in spec['transforms'].items():
            # Combine entities with query. Query values override file entities.
            if k.startswith('bold'):
                query = {**bold_filters, **entities.get('bold', {}), **q}
            elif k.startswith('anat'):
                query = {**anat_filters, **entities.get('anat', {}), **q}

            if k == 'boldref2fmap':
                query['to'] = fieldmap_id

            item = layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

    # Search for requested output spaces
    if spaces is not None:
        spaces_found, anat2outputspaces_xfm = [], []
        for space in spaces.references:
            # First try to find processed BOLD+mask files in the requested space
            anat2space_query = {
                **anat_filters,
                **entities.get('anat', {}),
                **spec['transforms']['anat2outputspaces_xfm'],
            }
            anat2space_query['to'] = space.space
            print(space)
            print(anat2space_query)
            item = layout.get(return_type='filename', **anat2space_query)
            anat2outputspaces_xfm.append(item[0] if item else None)
            spaces_found.append(bool(item))

        if all(spaces_found):
            derivs_cache['anat2outputspaces_xfm'] = anat2outputspaces_xfm
        else:
            missing_spaces = ', '.join(
                [s.space for s, found in zip(spaces.references, spaces_found) if not found]
            )
            raise ValueError(
                f'Transforms to the following requested spaces not found: {missing_spaces}.'
            )

    # Search for raw BOLD data
    if raw_dataset is not None:
        if isinstance(raw_dataset, Path):
            raw_layout = BIDSLayout(raw_dataset, config=['bids'], validate=False)
        else:
            raw_layout = raw_dataset

        for k, q in spec['raw'].items():
            # Combine entities with query. Query values override file entities.
            if k.startswith('bold'):
                query = {**bold_filters, **entities.get('bold', {}), **q}
            elif k.startswith('anat'):
                query = {**anat_filters, **entities.get('anat', {}), **q}

            print(k)
            print(query)
            item = raw_layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

    return derivs_cache


def write_bidsignore(deriv_dir):
    bids_ignore = (
        '*.html',
        'logs/',
        'figures/',  # Reports
        '*_xfm.*',  # Unspecified transform files
        '*.surf.gii',  # Unspecified structural outputs
        # Unspecified functional outputs
        '*_boldref.nii.gz',
        '*_bold.func.gii',
        '*_mixing.tsv',
        '*_timeseries.tsv',
    )
    ignore_file = Path(deriv_dir) / '.bidsignore'

    ignore_file.write_text('\n'.join(bids_ignore) + '\n')


def write_derivative_description(bids_dir, deriv_dir, dataset_links=None):
    import os

    from fmripost_phase import __version__

    DOWNLOAD_URL = f'https://github.com/nipreps/fmripost_phase/archive/{__version__}.tar.gz'

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': 'fMRIPost-AROMA- ICA-AROMA Postprocessing Outputs',
        'BIDSVersion': '1.9.0dev',
        'DatasetType': 'derivative',
        'GeneratedBy': [
            {
                'Name': 'fMRIPost-AROMA',
                'Version': __version__,
                'CodeURL': DOWNLOAD_URL,
            }
        ],
        'HowToAcknowledge': 'Please cite fMRIPost-AROMA when using these results.',
    }

    # Keys that can only be set by environment
    if 'FMRIPOST_AROMA_DOCKER_TAG' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'docker',
            'Tag': f'nipreps/fmriprep:{os.environ["FMRIPOST_AROMA__DOCKER_TAG"]}',
        }
    if 'FMRIPOST_AROMA__SINGULARITY_URL' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'singularity',
            'URI': os.getenv('FMRIPOST_AROMA__SINGULARITY_URL'),
        }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        orig_desc = json.loads(fname.read_text())

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasets'] = [
            {'URL': f'https://doi.org/{orig_desc["DatasetDOI"]}', 'DOI': orig_desc['DatasetDOI']}
        ]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    # Add DatasetLinks
    if dataset_links:
        desc['DatasetLinks'] = {k: str(v) for k, v in dataset_links.items()}
        if 'templateflow' in dataset_links:
            desc['DatasetLinks']['templateflow'] = 'https://github.com/templateflow/templateflow'

    Path.write_text(deriv_dir / 'dataset_description.json', json.dumps(desc, indent=4))
