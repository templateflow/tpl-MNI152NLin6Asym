import argparse
import logging
import os
from pathlib import Path
import typing as ty

import templateflow.api as tf
import nipype.pipeline.engine as pe
import niworkflows
from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration
from niworkflows.interfaces.nibabel import ApplyMask
from niworkflows.interfaces.reportlets.registration import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
from niworkflows.utils.spaces import OutputReferencesAction, Reference

HERE = Path(__file__).parent


def init_intertemplate_registration(
    moving_template: Reference,
    reference_template: Reference,
    outdir: str | Path,
    omp_nthreads: int = 1,
    init_affine: str | None = None,
    modality: str = 'T1w',
    quality: ty.Literal['testing', 'precise'] = 'precise',
    name: str = 'intertemplate_registration_{mov}_{ref}',
) -> pe.Workflow:
    """
    Perform image registration between two templates.
    """
    workflow = pe.Workflow(
        name=name.format(
            mov=moving_template.fullname.replace(':', ''),
            ref=reference_template.fullname.replace(':', ''),
        )
    )

    def get_templates(reference: Reference, modality: str) -> tuple[str, str]:
        tpl = reference.space
        spec = reference.spec

        # Ensure highest quality
        res = spec.pop('res', None)
        if res == 'native':
            res = 1
        spec['resolution'] = res or 1

        template = tf.get(tpl, raise_empty=True, desc=None, suffix=modality, **spec)

        # Testing MNIInfant+1
        mask = HERE / '..' / 'nibabies' / 'data' / 'tpl-MNIInfant_cohort-1_desc-reg_mask.nii.gz'
        # mask = tf.get(tpl, desc='brain', suffix='mask', **spec)
        if not mask:
            mask = tf.get(tpl, raise_empty=True, label='brain', suffix='mask', **spec)
        if not mask:
            raise FileNotFoundError(f'No mask found for {reference}')
        return str(template), str(mask)

    mov_tpl, mov_mask = get_templates(moving_template, modality)
    ref_tpl, ref_mask = get_templates(reference_template, modality)

    masked_mov = pe.Node(ApplyMask(in_file=mov_tpl, in_mask=mov_mask), name='mask_mov')
    masked_ref = pe.Node(ApplyMask(in_file=ref_tpl, in_mask=ref_mask), name='mask_ref')

    ants_env = {
        'NSLOTS': '%d' % omp_nthreads,
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '%d' % omp_nthreads,
        'OMP_NUM_THREADS': '%d' % omp_nthreads,
    }

    reg_opts = niworkflows.data.load(f't1w-mni_registration_{quality}_000.json')
    reg = pe.Node(
        Registration(from_file=reg_opts),
        name='reg',
        n_procs=omp_nthreads,
    )
    reg.config = {'execution': {'remove_unnecessary_outputs': False}}

    reg_rpt = pe.Node(
        SimpleBeforeAfter(
            before_label=reference_template.fullname,
            after_label=f'{moving_template.fullname}->{reference_template.fullname}',
        ),
        name='reg_rpt',
        mem_gb=0.1,
    )
    reg_rpt.inputs.before = ref_tpl

    workflow.connect([
        (masked_mov, reg, [('out_file', 'moving_image')]),
        (masked_ref, reg, [('out_file', 'fixed_image')]),
        (reg, reg_rpt, [('warped_image', 'after')]),
    ])  # fmt:off

    if not init_affine:
        from nipype.interfaces.ants.utils import AI

        init_aff = pe.Node(
            AI(
                metric=('Mattes', 32, 'Regular', 0.2),
                transform=('Affine', 0.1),
                search_factor=(20, 0.12),
                principal_axes=False,
                convergence=(10, 1e-6, 10),
                verbose=True,
                fixed_image=ref_tpl,
                fixed_image_mask=ref_mask,
                moving_image=mov_tpl,
                moving_image_mask=mov_mask,
                environ=ants_env,
            ),
            name='init_aff',
            n_procs=omp_nthreads
        )

        workflow.connect([
            (init_aff, reg, [('output_transform', 'initial_moving_transform')]),
        ])  # fmt:off
    else:
        reg.inputs.initial_moving_transform = init_affine

    return workflow



def main(argv=None):
    """Run template to template registration."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mov', nargs="*", action=OutputReferencesAction, help='the moving template')
    parser.add_argument('--ref', nargs="*", action=OutputReferencesAction, help='the reference template')
    parser.add_argument('--nprocs', default=os.cpu_count(), type=int, help='Number of CPUs to use.')
    parser.add_argument('--init-affine', help='File to initialize to reference template affine')
    parser.add_argument('--modality', default='T1w', help='Template modality to use')
    parser.add_argument('--quick', action='store_true', help='Quick (sloppy) registration')
    parser.add_argument('--workdir', default=Path('.'), help='Path to save workflow')
    parser.add_argument('--outdir', default=Path('.'), help='Path to save outputs')
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbose_count',
        action='count',
        default=0,
        help='Increases log verbosity for each occurence, debug level is -vvv'
    )
    pargs = parser.parse_args(argv)
    moving = pargs.mov.references[0]
    reference = pargs.ref.references[0]
    init_affine = pargs.init_affine
    if init_affine:
        init_affine = str(Path(pargs.init_affine).absolute())

    print(f"{moving=}\n{reference=}\n{init_affine=}")

    workflow = init_intertemplate_registration(
        moving_template=moving,
        reference_template=reference,
        omp_nthreads=pargs.nprocs,
        init_affine=init_affine,
        modality=pargs.modality,
        quality='testing' if pargs.quick else 'precise',
        outdir = pargs.outdir,
    )

    workdir = pargs.workdir
    workflow.base_dir = workdir

    # Retrieve logging level
    log_level = int(max(25 - 5 * pargs.verbose_count, logging.DEBUG))
    # Set logging
    logging.getLogger('nipype.workflow').setLevel(log_level)
    logging.getLogger('nipype.interface').setLevel(log_level)
    logging.getLogger('nipype.utils').setLevel(log_level)

    workflow.run()


if __name__ == "__main__":
    main()
