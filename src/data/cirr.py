import json
import random
from pathlib import Path
import logging
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.data.transforms import transform_test, transform_train
from src.data.utils import pre_caption

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning
import os
import pickle
import PIL
def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class CIRRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        img_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = CIRR(
            transform=self.transform_train,
            path=annotation,
        )
        self.data_val = self.data_train

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def train_dataloader_shuffle(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
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


class CIRRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        img_dirs: str,
        emb_dirs: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        split: str = "test",
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform_test = transform_test(image_size)

        self.data_test = CIRR(
            transform=self.transform_test,
            path=annotation,
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


class CIRRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
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

        self.pairid2ref = {ann["pairid"]: ann["reference"] for ann in self.annotation}
        self.pairid2members = {
            ann["pairid"]: ann["img_set"]["members"] for ann in self.annotation
        }
        if "test" not in Path(self.annotation_pth).stem:
            self.pairid2tar = {
                ann["pairid"]: ann["target_hard"] for ann in self.annotation
            }
        else:
            self.pairid2tar = None

        if split == "train":
            img_pths = self.img_dir.glob("*/*.png")
        else:
            img_pths = self.img_dir.glob("*.png")
        self.id2imgpth = {img_pth.stem: img_pth for img_pth in img_pths}
        for ann in self.annotation:
            assert (
                ann["reference"] in self.id2imgpth
            ), f"Path to reference {ann['reference']} not found in {self.img_dir}"
            if split != "test":
                assert (
                    ann["target_hard"] in self.id2imgpth
                ), f"Path to target {ann['target_hard']} not found"

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        reference_img_pth = self.id2imgpth[ann["reference"]]
        reference_img = Image.open(reference_img_pth).convert("RGB")
        reference_img = self.transform(reference_img)

        caption = pre_caption(ann["caption"], self.max_words)

        if self.split == "test":
            return {
                "ref_img": reference_img,
                "edit": caption,
                "pair_id": ann["pairid"],
                "ref_webvid_caption": "",
            }

        target_img_pth = self.id2imgpth[ann["target_hard"]]
        target_img = Image.open(target_img_pth).convert("RGB")
        target_img = self.transform(target_img)


        return_dict = {
            "ref_img": reference_img,
            "tar_img": target_img,
            "edit": caption,
            "pair_id": ann["pairid"],
            "ref_webvid_caption": "",
            "tag_webvid_caption": "",
        }

        return return_dict


class CIRR(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, case_look=False) -> None:
        super(CIRR, self).__init__()
        self.path = path
        self.caption_dir = self.path + 'captions'
        self.split_dir = self.path + 'image_splits'
        self.transform = transform
        self.case_look = case_look
        # train data
        with open(os.path.join(self.caption_dir, "cap.rc2.train.json"), 'r') as f:
            self.cirr_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys())

        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            self.val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            self.val_image_path = json.load(f)
            self.val_image_name = list(self.val_image_path.keys())
        # val data
        if not os.path.exists(os.path.join(self.path, 'cirr_val_queries.pkl')):
            self.val_queries, self.val_targets = self.get_val_queries()
            save_obj(self.val_queries, os.path.join(self.path, 'cirr_val_queries.pkl'))
            save_obj(self.val_targets, os.path.join(self.path, 'cirr_val_targets.pkl'))
        else:
            self.val_queries = load_obj(os.path.join(self.path, 'cirr_val_queries.pkl'))
            self.val_targets = load_obj(os.path.join(self.path, 'cirr_val_targets.pkl'))
        # test data
        if not os.path.exists(os.path.join(self.path, 'cirr_test_queries.pkl')):
            self.test_name_list, self.test_img_data, self.test_queries = self.get_test_queries()
            save_obj(self.test_name_list, os.path.join(self.path, 'cirr_test_name_list.pkl'))
            save_obj(self.test_img_data, os.path.join(self.path, 'cirr_test_img_data.pkl'))
            save_obj(self.test_queries, os.path.join(self.path, 'cirr_test_queries.pkl'))
        else:
            self.test_name_list = load_obj(os.path.join(self.path, 'cirr_test_name_list.pkl'))
            self.test_img_data = load_obj(os.path.join(self.path, 'cirr_test_img_data.pkl'))
            self.test_queries = load_obj(os.path.join(self.path, 'cirr_test_queries.pkl'))

        

    def __len__(self):
        return len(self.cirr_data)

    def __getitem__(self, idx):
        caption = self.cirr_data[idx]
        reference_name = caption['reference']
        mod_str = caption['caption']
        target_name = caption['target_hard']

        out = {
            "ref_img": self.get_img(self.train_image_path[reference_name]),
            "tar_img": self.get_img(self.train_image_path[target_name]),
            "edit": mod_str,
            "ref_description": "",
            "tag_description": "",
            "ref_webvid_caption": "",
            "tag_webvid_caption": "",
        }

        return out

    def get_img(self, img_path, return_raw=False):
        img_path = os.path.join(self.path, img_path.lstrip('./'))
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
            
        if return_raw:
            transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
            return transform(img)

        if self.transform:
            #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            img = self.transform(img)
        return img

    def get_val_queries(self):
        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = list(val_image_path.keys())
        
        test_queries = []
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['caption']
            reference_name = caption['reference']
            target_name = caption['target_hard']
            subset_names = caption['img_set']['members']
            subset_ids = [val_image_name.index(n) for n in subset_names]

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_img(val_image_path[reference_name])
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_img(val_image_path[target_name])
            out['mod'] = {'str':mod_str}
            out['subset_id'] = subset_ids
            if self.case_look:
                out['raw_src_img_data'] = self.get_img(val_image_path[reference_name], return_raw=True)
                out['raw_tag_img_data'] = self.get_img(val_image_path[target_name], return_raw=True)
            
            test_queries.append(out)

        test_targets = []
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(val_image_path[name])
            if self.case_look:
                out['raw_tag_img_data'] = self.get_img(val_image_path[name], return_raw=True)
            test_targets.append(out)

        return test_queries, test_targets
    
    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        for i in range(len(test_data)):
            out = {}
            caption = test_data[i]
            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_img(test_image_path[caption['reference']])
            out['reference_name'] = caption['reference']
            out['mod'] = caption['caption']
            out['subset'] = caption['img_set']['members']
            queries.append(out)

        image_name = []
        image_data = []
        for i in range(len(test_image_name)):
            name = test_image_name[i]
            data = self.get_img(test_image_path[name])
            image_name.append(name)
            image_data.append(data)
        return image_name, image_data, queries



