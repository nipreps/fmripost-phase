.. include:: links.rst

===========================
Processing pipeline details
===========================

*fMRIPost-Phase* adapts its pipeline depending on what data and metadata are
available and are used as the input.
For example, slice timing correction will be
performed only if the ``SliceTiming`` metadata field is found for the input
dataset.

A (very) high-level view of the simplest pipeline is presented below:

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmripost_phase.workflows.tests import mock_config
    from fmripost_phase.workflows.base import init_single_subject_wf

    with mock_config():
        wf = init_single_subject_wf('01')


**********
Input Data
**********

fMRIPost-Phase expects both a raw BIDS dataset and fMRIPrep derivatives as input.
The raw BIDS dataset should contain both magnitude and phase information for BOLD runs.
Optionally, the raw data may also include noise scans (``_noRF``) for the magnitude and phase data.

The fMRIPrep derivatives should include images and transforms needed to process the raw data,
including::

    - Motion correction transforms
    - BOLD reference-to-fieldmap transform
    - The BOLD-to-anatomical transform
    - Anatomical-to-standard space transforms, as requested
    - Brain mask in BOLD reference space
    - Confounds TSV file


*********************
Thermal Noise Removal
*********************

fMRIPost-Phase supports the removal of thermal noise from the magnitude and phase data via dwidenoise.


***********************
Slice Timing Correction
***********************

fMRIPost-Phase will apply slice-timing correction to the magnitude data.


****************
Phase Unwrapping
****************

fMRIPost-Phase will unwrap the phase data.


**********************************
Resampling to BOLD Reference Space
**********************************

fMRIPost-Phase will resample the magnitude and phase data to the BOLD reference space.


*********
RETROICOR
*********


****************
Phase Regression
****************


*******************
Phase Jolt and Jump
*******************


**************************************
Complex Independent Component Analysis
**************************************


*******************
Confound Extraction
*******************
