### img_flist.py
# A dataset class to load image file list from a txt/csv file.
# Author: Gina Wu @ 01/22
###

import os
import torch.utils.data as data
from PIL import Image
import json
from typing import List, Dict, Any

_VALID_DOMAIN = ["usa", "asia"]
_VALID_SPLIT = ["train", "test"]
_DEFAULT_STR = "NULL"

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_ann(imid):
    return {
        "image_id":imid,
        "category": int(0),
        "class_name": _DEFAULT_STR
    }

def default_meta(imid):
    return {
        "image_id":imid,
        "caption": _DEFAULT_STR,
        "tags"       : _DEFAULT_STR,
        "url"        : _DEFAULT_STR
    }

def default_loc(imid):
    return {
        "image_id": imid,
        "countryRegionIso2": _DEFAULT_STR,
        "continentRegion"  : _DEFAULT_STR,
        "latitude"         : 0.0,
        "longitude"        : 0.0
    }


class ImageJSONLoader(data.Dataset):

    def __init__(self, root_dir, json_path, domain="usa", split="train", transform=None, target_transform=None,
                 loader=default_loader, return_ann=True, return_loc=False, return_meta=False, 
                 _loc_keys=None, _meta_keys=None):

        if _loc_keys is not None:
            assert isinstance(_loc_keys, list), "loc keys has to be a list."
            assert return_loc
            _loc_keys += list(default_loc(0).keys())
        else:
            _loc_keys = list(default_loc(0).keys())
        _loc_keys = list(set(_loc_keys))
        
        if _meta_keys is not None:
            assert isinstance(_meta_keys, list), "meta keys has to be a list."
            assert return_meta
            _meta_keys += list(default_meta(0).keys())
        else:
            _meta_keys = list(default_meta(0).keys())
        _meta_keys = list(set(_meta_keys))

        if not isinstance(domain, list):
            domain = [domain]
        if not isinstance(split, list):
            split = [split]

        assert all([d in _VALID_DOMAIN for d in domain]), "Invalid Domain".format(domain)
        assert all([s in _VALID_SPLIT for s in split]), "split has to be {}. {} not recognized".format("|".join(_VALID_SPLIT), split)

        self.root_dir = root_dir
        keytag = []
        for d in domain:
            for s in split:
                keytag.append("{}_{}".format(d, s))
        json_data = json.load(open(json_path))

        self.return_ann = return_ann
        self.return_loc = return_loc
        self.return_meta = return_meta

        self.info = json_data['info']
        self.category_mapping = json_data['categories']

        self.classname_to_id = {c["category_name"]:int(c["category_id"]) for c in self.category_mapping}
        self.id_to_classname = {v:k for k,v in self.classname_to_id.items()}
        
        imdata = [json_data[kt] for kt in keytag]
        id_to_im = {im["id"]:im for imd in imdata for im in imd["images"]}

        id_to_ann = {image_id:default_ann(image_id) for image_id in id_to_im.keys()}
        id_to_loc = {image_id:default_loc(image_id) for image_id in id_to_im.keys()}
        id_to_meta = {image_id:default_meta(image_id) for image_id in id_to_im.keys()}

        if return_ann:
            id_to_ann = {ann["image_id"]:ann for imd in imdata for ann in imd["annotations"]}
            assert len(id_to_ann) == len(id_to_im), "Annotations Missing"

        if return_loc:
            id_to_loc = {loc["image_id"]:loc for imd in imdata for loc in imd["locations"]}
            assert len(id_to_loc) == len(id_to_im), "Locations Missing"

        if return_meta:
            id_to_meta = {meta["image_id"]:meta for imd in imdata for meta in imd["metadata"]}
            assert len(id_to_meta) == len(id_to_im), "Metadata Missing"

        ## combine image, annotation and locations
        self.geodata = []
        for imid in id_to_im.keys():

            self.geodata.append((
                imid,
                id_to_im[imid]["filename"],
                id_to_ann[imid]["category"],
                {k:id_to_loc[imid].get(k,"NULL") for k in _loc_keys},
                {k:id_to_meta[imid][k] for k in _meta_keys}
            ))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        imid, impath, target, location, metadata = self.geodata[index]
        location.pop("image_id",None)
        metadata.pop("image_id",None)

        impath = os.path.join(self.root_dir, impath)
        img = self.loader(impath)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return_obj = [imid, img]

        if self.return_ann:
            return_obj.append(target)

        if self.return_loc:
            return_obj.append(location)
        
        if self.return_meta:
            return_obj.append(metadata)

        return return_obj


    def __len__(self):
        return len(self.geodata)

if __name__ == "__main__":

    # places_root = "/newfoundland2/tarun/datasets/Places205/data/vision/torralba/deeplearning/GeoDA/GeoPlaces_dataset/"
    # places_json = "/home/tarun/GeoDA_Project/metadata/geoPlaces_metadata.json"
    imnet_root = "/newdata/tarun/datasets/WebVision/GeoILSVRC"
    imnet_json = "/home/tarun/GeoDA_Project/metadata/geoImnet_metadata.json"
    dataset = ImageJSONLoader(imnet_root, imnet_json, domain=["asia", "usa"], split=["train", "test"])

    print(len(dataset))

