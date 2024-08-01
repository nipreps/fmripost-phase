# -*- coding: utf-8 -*-
"""LayNii is a C++ library for layer fMRI processing and analysis."""

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class _LayNiiPhaseJoltInputSpec(CommandLineInputSpec):
    """Input specification for LN2_PHASE_JOLT."""

    in_file = File(
        exists=True,
        position=0,
        argstr="-input %s",
        mandatory=True,
        desc=(
            "Nifti image that will be used to compute gradients. "
            "This can be a 4D nifti. "
            "In 4D case, 3D gradients will be computed for each volume."
        ),
    )
    int13 = traits.Bool(
        False,
        argstr="-int13",
        mandatory=False,
        usedefault=True,
        desc=(
            "Cast the input range from [-4096 4096] to [0 2*pi]. "
            "This option is often needed with Siemens phase images as they "
            "commonly appear to be uint12 range with scl_slope = 2, and "
            "scl_inter = -4096 in the header. "
            "Meaning that the intended range is int13, "
            "even though the data type is uint16 and only int12 portion "
            "is used to store the phase values."
        ),
    )
    phase_jump = traits.Bool(
        False,
        argstr="-phase_jump",
        mandatory=False,
        usedefault=True,
        desc="Output L1 norm of the 1st spatial derivative.",
    )
    twod = traits.Bool(
        False,
        argstr="-2D",
        mandatory=False,
        usedefault=True,
        desc="Do not compute along z. Experimental.",
    )
    output = traits.Str(
        argstr="-output %s",
        desc="Output basename for all outputs.",
    )
    debug = traits.Bool(
        False,
        argstr="-debug",
        mandatory=False,
        usedefault=True,
        desc="Save extra intermediate outputs.",
    )


class _LayNiiPhaseJoltOutputSpec(TraitedSpec):
    """Output specification for C3d."""

    out_file = File(exists=True)


class LayNiiPhaseJolt(CommandLine):
    """L1 norm of phase image second spatial derivatives.

    Uses 1-jump voxel neighbors for computations.
    """

    input_spec = _LayNiiPhaseJoltInputSpec
    output_spec = _LayNiiPhaseJoltOutputSpec

    _cmd = "LN2_PHASE_JOLT"
