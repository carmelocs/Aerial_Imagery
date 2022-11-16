# Extract image regions defined in a VIA project
#
# Author: Abhishek Dutta <adutta _AT_ robots.ox.ac.uk>
# Date: 19 Nov. 2019
#

# command to execute (with parameters): 
# python extract_img_regions.py --project ./via_project_12Nov2022_21h0m.json --imdir ./split_images --cropdir ./output --cropmeta ./output/cropmeta.csv 

# command to debug: 
# python -m pdb extract_img_regions.py --project ./via_project_12Nov2022_21h0m.json --imdir ./split_images --cropdir ./output --cropmeta ./output/cropmeta.csv 

import json
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument("--project", help="location of a VIA 2.x.y project saved as a JSON file")
parser.add_argument("--imdir", help="location of images referenced in the VIA project")
parser.add_argument("--cropdir", help="cropped images are stored in this folder")
parser.add_argument("--cropmeta", help="metadata associated with each cropped image gets saved to this file")
parser.add_argument("--croppad", help="padding (default=0) applied to each cropped image (to provide additional context)", type=int, default=0)
args = parser.parse_args()

via = {}
with open(args.project, 'r') as f:
    via = json.load(f)

label_names = {"good": 0, "damaged": 1}

cropmeta = open(args.cropmeta, 'w')
cropmeta.write('file_name,metadata,label\n')
for fid in via['_via_img_metadata']:
    fn = os.path.join(args.imdir, via['_via_img_metadata'][fid]['filename'])
    if not os.path.isfile(fn):
        print('File not found! %s' %(fn))
        continue
    im = Image.open(fn)
    imwidth, imheight = im.size
    # rindex = 1
    for rindex, region in enumerate(via['_via_img_metadata'][fid]['regions']):
        if region['shape_attributes']['name'] != 'rect':
            print('extraction of %s regions not yet implemented!' % region['shape_attributes']['name'])
            continue
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        w = region['shape_attributes']['width']
        h = region['shape_attributes']['height']

        left = max(0, x - args.croppad)
        top = max(0, y - args.croppad)
        right = min(imwidth, x + w + args.croppad)
        bottom = min(imheight, y + h + args.croppad)
        crop = im.crop((left, top, right, bottom))
        crop = crop.resize((224, 224))
        extold = os.path.splitext(via['_via_img_metadata'][fid]['filename'])[1]
        extnew = extold.replace('.', '_' + str(rindex) + '.')
        cropname = via['_via_img_metadata'][fid]['filename'].replace(extold, extnew)
        print(cropname)
        cropfn = os.path.join(args.cropdir, cropname)
        crop.save(cropfn)
        cropmeta.write('%s,"%s",%s\n' %(cropname,
                                      str(region['region_attributes']),
                                      label_names[region['region_attributes']["type"]]))
cropmeta.close()
