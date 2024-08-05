import glob
import json
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset, Subset


class PointNetFeat(nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 512):
        super(PointNetFeat, self).__init__()
        self.conv1 = nn.Conv1d(in_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_features, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        out4 = torch.max(out3, dim=-1)[0]
        return out4


class PointNet(nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 512, num_classes: int = 10):
        super(PointNet, self).__init__()
        # determine whether to fuse per-point features
        self.feat_net = PointNetFeat(in_features=in_features, out_features=out_features)
        # classification head
        self.fc1 = nn.Linear(out_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, pointcloud: torch.Tensor):
        """shape of input: [num_samples, 3, num_points]"""
        global_features = self.feat_net(pointcloud)
        # only predict class for first point cloud
        pred = F.relu(self.bn1(self.fc1(global_features)))
        pred = F.relu(self.bn2(self.dropout(self.fc2(pred))))
        pred = self.fc3(pred)
        pred = F.log_softmax(pred, dim=1)
        return pred

    def extract_features(self, pointcloud: torch.Tensor):
        """shape of input: [num_samples, 3, num_points]"""
        features = self.feat_net(pointcloud)
        features = F.relu(self.bn1(self.fc1(features)))
        features = self.fc2(features)
        return features


class OakInkPointDataset(Dataset):
    def __init__(self, num_points: int = 1024):
        super().__init__()

        statistics = []
        for filepath in glob.glob("data/oakink_shadow_dataset_valid_new/*/*.json", recursive=True):
            with open(filepath) as f:
                data = json.load(f)
            statistics.append({"category": data["category"], "code": data["object_code"]})
        statistics = pd.DataFrame(statistics).drop_duplicates().reset_index(drop=True)
        statistics["category"] = statistics["category"].astype("category").cat.codes

        filepaths = []
        categories = []
        codes = []
        for _, item in statistics.iterrows():
            filepath = Path("assets") / "oakink" / item["code"] / "align" / "decomposed.obj"
            if not filepath.exists():
                continue
            filepaths.append(filepath)
            categories.append(item["category"])
            codes.append(item["code"])

        meshes = load_objs_as_meshes(filepaths, load_textures=False, device="cpu")
        self.pointclouds = sample_points_from_meshes(meshes, num_samples=num_points)
        self.pointclouds = torch.einsum("b n d -> b d n", self.pointclouds)
        self.categories = torch.tensor(categories, dtype=torch.long)
        self.codes = codes

        self.num_categories = int(torch.max(self.categories) + 1)
        self.num_points = num_points
        self.num_samples = self.categories.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index) -> dict:
        return {
            "pointcloud": self.pointclouds[index],
            "category": self.categories[index],
        }


class PointNetPretrain(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self._set_hparams(hparams)
        self.model = PointNet(
            in_features=hparams["in_features"],
            out_features=hparams["out_features"],
            num_classes=hparams["num_classes"],
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["pointcloud"], batch["category"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["pointcloud"], batch["category"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = (torch.argmax(y_hat, dim=1) == y).float().mean()
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("val_accuracy", avg_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    in_features: int = 3
    hidden_features: int = 128
    out_features: int = 512
    learning_rate: float = 5e-4

    seed_everything(42)
    dataset = OakInkPointDataset(num_points=512)

    indices = torch.randperm(len(dataset))
    train_indices = indices[: int(0.8 * len(indices))]
    valid_indices = indices[int(0.8 * len(indices)) :]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, num_workers=4, shuffle=False)

    hparams = {
        "in_features": in_features,
        "out_features": out_features,
        "num_classes": dataset.num_categories,
        "learning_rate": learning_rate,
    }
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=20, mode="max")
    trainer = pl.Trainer(gpus=1, min_epochs=30, max_epochs=100, log_every_n_steps=1, callbacks=[early_stopping])
    model = PointNetPretrain(hparams)
    trainer.fit(model, train_loader, valid_loader)

    # save model to file
    torch.save(model.model.state_dict(), "pointnet_pretrain.pt")

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["pointcloud"], batch["category"]
            hidden = model.model.extract_features(x)
            embeddings.append(hidden)
    embeddings = torch.cat(embeddings, dim=0)

    frame = pd.DataFrame(embeddings.numpy(), index=dataset.codes, columns=[f"feat_{i}" for i in range(hidden_features)])
    frame.index.rename("code", inplace=True)
    frame.to_csv("pointnet_pretrain_embeddings.csv")
