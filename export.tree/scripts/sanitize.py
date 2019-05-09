import nibabel as nb
from pathlib import Path

def sanitize(input_fname):
    im = nb.as_closest_canonical(nb.load(str(input_fname)))
    hdr = im.header.copy()
    dtype = 'int16'
    slope = 1.0
    inter = 0.0
    if str(input_fname).endswith('_mask.nii.gz'):
        dtype = 'uint8'

    if 'atlas-HO' in str(input_fname):
        dtype = 'uint8'

        if str(input_fname).endswith('atlas-HOSPA_probseg.nii.gz'):
            dtype = 'float32'

        if str(input_fname).endswith('_probseg.nii.gz'):
            slope = 0.01

    hdr.set_data_dtype(dtype)
    nii = nb.Nifti1Image(im.get_data().astype(dtype), im.affine, hdr)

    sform = nii.header.get_sform()
    nii.header.set_sform(sform, 4)
    nii.header.set_qform(sform, 4)
    nii.header.set_slope_inter(slope=slope, inter=inter)
    nii.header.set_xyzt_units(xyz='mm')
    print(nii.header['scl_slope'])
    nii.to_filename(str(input_fname))

for f in Path().glob('tpl-*_atlas-HO*.nii.gz'):
    print('Sanitizing file %s' % f)
    sanitize(f)

