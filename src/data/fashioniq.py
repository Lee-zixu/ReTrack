import json
from pathlib import Path

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import transform_test, transform_train
from src.data.utils import pre_caption

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning
import string 
import os
import pickle
import PIL
import logging
import random

class FashionIQDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        targets: dict = {"train": "", "val": ""},
        img_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        category: str = "dress",
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)
        self.data_train = FashionIQ(
            transform=self.transform_train,
            path=annotation,
            category=category
        )
        self.data_val = self.data_train
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class FashionIQTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        annotation_cap: str,
        targets: str,
        img_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        category: str = "dress",
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_test = transform_test(image_size)

        self.data_test = FashionIQ(
            transform=self.transform_test,
            path=annotation,
            category=category
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class FashionIQDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        annotation_cap: str,
        targets: str,
        img_dir: str,
        emb_dir: str,
        split: str,
        max_words: int = 30,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.annotation_pth = annotation
        assert Path(annotation).exists(), f"Annotation file {annotation} does not exist"
        self.annotation = json.load(open(annotation, "r"))
        assert Path(annotation_cap).exists(), f"Annotation file {annotation_cap} does not exist"
        self.annotation_cap = json.load(open(annotation_cap, "r"))
        assert Path(targets).exists(), f"Targets file {targets} does not exist"
        # self.targets = json.load(open(targets, "r"))
        # self.target_ids = list(set(self.targets))

        
        with open("/root/autodl-tmp/fashion_iq_data/captions/correction_dict_all.json", 'r') as f:
            self.correction_dict = json.load(f)
        self.target_ids = []
        for ann in self.annotation:
            if ann['candidate'] not in self.target_ids:
                self.target_ids.append(ann["candidate"])
            if ann['target'] not in self.target_ids:
                self.target_ids.append(ann["target"])


        self.target_ids.sort()

        self.split = split
        self.max_words = max_words
        self.img_dir = Path(img_dir)
        # self.emb_dir = Path(emb_dir)
        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        assert self.img_dir.exists(), f"Image directory {img_dir} does not exist"
        # assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"

        self.id2int = {id: i for i, id in enumerate(self.target_ids)}
        self.int2id = {i: id for i, id in enumerate(self.target_ids)}

        self.pairid2ref = {
            id: self.id2int[ann["candidate"]] for id, ann in enumerate(self.annotation)
        }
        self.pairid2tar = {
            id: self.id2int[ann["target"]] for id, ann in enumerate(self.annotation)
        }

        img_pths = self.img_dir.glob("*.jpg")
        # emb_pths = self.emb_dir.glob("*.pth")
        self.id2imgpth = {img_pth.stem: img_pth for img_pth in img_pths}
        # self.id2embpth = {emb_pth.stem: emb_pth for emb_pth in emb_pths}

        for ann in self.annotation:
            assert (
                ann["candidate"] in self.id2imgpth
            ), f"Path to image candidate {ann['candidate']} not found in {self.img_dir}"
            # assert (
            #     ann["candidate"] in self.id2embpth
            # ), f"Path to embedding candidate {ann['candidate']} not found in {self.emb_dir}"
            
            assert (
                ann["target"] in self.id2imgpth
            ), f"Path to image target {ann['target']} not found"
            # assert (
            #     ann["target"] in self.id2embpth
            # ), f"Path to embedding target {ann['target']} not found"

    def correct_text(self, text, correction_dict):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

        return text

    def concat_text(self, captions, correction_dict):
        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
        return text

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        reference_img_pth = self.id2imgpth[ann["candidate"]]
        reference_img = Image.open(reference_img_pth).convert("RGB")
        reference_img = self.transform(reference_img)

        # cap1, cap2 = ann["captions"]
        caption = self.concat_text(ann['captions'], self.correction_dict)  # f"{cap1} and {cap2}"
        # caption = pre_caption(caption, self.max_words)

        target_img_pth = self.id2imgpth[ann["target"]]
        target_img = Image.open(target_img_pth).convert("RGB")
        target_img = self.transform(target_img)

        # print(self.annotation_cap[ann["candidate"]])
        return {
            "ref_img": reference_img,
            "tar_img": target_img,
            "edit": caption,
            "pair_id": index,
            "ref_webvid_caption": "", #self.annotation_cap[ann["candidate"]],
            "tag_webvid_caption": "" #self.annotation_cap[ann["target"]],
        }


class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, category, transform=None, split='val-split'):
        super().__init__()

        self.path = path
        self.category = category
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        self.split = split

        self.train_data = []
        self.train_init_process()

        if category != "all":
            self.test_queries, self.test_targets = self.get_test_data()

    def train_init_process(self):
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.category, 'train')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(self.category)), 'r') as f:
            correction_dict = json.load(f)
        for triplets in ref_captions:
            ref_id = triplets['candidate']
            tag_id = triplets['target']
            cap = self.concat_text(triplets['captions'], correction_dict)
            self.train_data.append({
                'target': self.category + '_' + tag_id,
                'candidate': self.category + '_' + ref_id,
                'captions': cap
            })




    def correct_text(self, text, correction_dict):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

        return text

    def concat_text(self, captions, correction_dict):
        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
        return text

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        caption = self.train_data[idx]
        # mod_str = self.concat_text(caption['captions'])
        mod_str = caption['captions']
        candidate = caption['candidate']
        target = caption['target']
        out = {
            "ref_img": self.get_img(candidate, stage=0),
            "tar_img": self.get_img(target, stage=0),
            "edit": mod_str,
            "ref_description": "",
            "tag_description": "",
            "ref_webvid_caption": "",
            "tag_webvid_caption": "",
        }
        return out

    def get_img(self, image_name, stage=0):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        return img


    def get_test_data(self):
        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(self.category, 'val')), 'r') as f:
            images = json.load(f)
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.category, 'val')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(self.category)), 'r') as f:
            correction_dict = json.load(f)


        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = self.concat_text(caption['captions'], correction_dict)
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = images.index(candidate)
            out['source_img_data'] = self.get_img(self.category + '_' + candidate, stage=1)
            out['target_img_id'] = images.index(target)
            out['target_img_data'] = self.get_img(self.category + '_' + target, stage=1)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)

        test_targets_id = []
        test_targets = []
        if self.split == 'val-split':
            for i in test_queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])
            
            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_img(self.category + '_' + images[i], stage=1)      
                test_targets.append(out)
        elif self.split == 'original-split':
            for id, image_name in enumerate(images):
                test_targets_id.append(id)
                out = {}
                out['target_img_id'] = id
                out['target_img_data'] = self.get_img(self.category + '_' + image_name, stage=1)      
                test_targets.append(out)
        return test_queries, test_targets

