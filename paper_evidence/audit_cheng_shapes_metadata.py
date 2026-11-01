import os
import io
import json
import zipfile
import numpy as np
import h5py
from scipy.io import loadmat

with zipfile.ZipFile('1512427.zip') as outer:
    inner_parts = [n for n in outer.namelist() if n.endswith('.zip')]
    print("Inner parts:", inner_parts)
    
    unique_shapes = set()
    all_fields = set()
    attributes_found = {}
    sample_count = 0
    
    for part in inner_parts:
        part_bytes = outer.read(part)
        with zipfile.ZipFile(io.BytesIO(part_bytes)) as inner:
            mat_files = [m for m in inner.namelist() if m.endswith('.mat')]
            # Check 50 from each part
            for mat_name in mat_files[:50]:
                mat_bytes = inner.read(mat_name)
                sample_count += 1
                if b"\x89HDF\r\n\x1a\n" in mat_bytes[:4096]:
                    with h5py.File(io.BytesIO(mat_bytes), 'r') as h5:
                        rec = h5['cjdata']
                        fields = list(rec.keys())
                        all_fields.update(fields)
                        # Check attributes
                        for k in rec.attrs:
                            attributes_found[k] = str(rec.attrs[k])
                        img = np.asarray(rec['image'][()])
                        unique_shapes.add(img.shape)
                else:
                    contents = loadmat(io.BytesIO(mat_bytes), squeeze_me=True, struct_as_record=False)
                    rec = contents['cjdata']
                    fields = dir(rec)
                    all_fields.update([f for f in fields if not f.startswith('_')])
                    img = getattr(rec, 'image')
                    unique_shapes.add(img.shape)

result = {
    "sample_files_checked": sample_count,
    "unique_image_shapes": [list(s) for s in unique_shapes],
    "cjdata_fields": sorted(list(all_fields)),
    "attributes_found": attributes_found,
    "plane_metadata_in_mat": "None. cjdata only contains PID, image, label, tumorBorder, tumorMask. No acquisition plane/view angle is stored in the .mat files."
}

print(json.dumps(result, indent=2))
with open('paper_evidence/cheng_metadata_audit.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Saved report to paper_evidence/cheng_metadata_audit.json")

