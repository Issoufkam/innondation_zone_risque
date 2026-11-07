"""Entraîne le U-Net léger sur CPU. Calcule des poids de classe (le fond
domine largement) et sauvegarde le meilleur modèle selon la perte de
validation.
"""

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, REPORTS_DIR
from dataset import VignettesInondation
from unet import UNetLeger

IGNORE_LABEL = 255
N_CLASSES = 3


def poids_classes(dataset, max_echantillons=200):
    compte = np.zeros(N_CLASSES, dtype="int64")
    pas = max(1, len(dataset) // max_echantillons)
    for i in range(0, len(dataset), pas):
        _, label = dataset[i]
        label = label.numpy()
        for c in range(N_CLASSES):
            compte[c] += (label == c).sum()
    compte = np.maximum(compte, 1)
    poids = compte.sum() / (N_CLASSES * compte)
    return torch.tensor(poids, dtype=torch.float32)


def boucle(modele, loader, criterion, optimizer=None):
    entrainement = optimizer is not None
    modele.train(entrainement)
    total_perte, n = 0.0, 0
    for bands, label in loader:
        if entrainement:
            optimizer.zero_grad()
        sortie = modele(bands)
        perte = criterion(sortie, label)
        if entrainement:
            perte.backward()
            optimizer.step()
        total_perte += perte.item() * bands.size(0)
        n += bands.size(0)
    return total_perte / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)

    train_ds = VignettesInondation("train")
    val_ds = VignettesInondation("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    in_channels = train_ds[0][0].shape[0]
    modele = UNetLeger(in_channels=in_channels, n_classes=N_CLASSES)

    print("Calcul des poids de classe (fond très majoritaire)...")
    poids = poids_classes(train_ds)
    print(f"Poids [fond, bâti, eau] = {poids.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(weight=poids, ignore_index=IGNORE_LABEL)
    optimizer = torch.optim.Adam(modele.parameters(), lr=args.lr)

    historique = []
    meilleure_perte = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perte_train = boucle(modele, train_loader, criterion, optimizer)
        perte_val = boucle(modele, val_loader, criterion)
        duree = time.time() - t0
        print(f"époque {epoch}/{args.epochs} - perte train {perte_train:.4f} - perte val {perte_val:.4f} ({duree:.0f}s)")
        historique.append({"epoch": epoch, "perte_train": perte_train, "perte_val": perte_val})

        if perte_val < meilleure_perte:
            meilleure_perte = perte_val
            torch.save(
                {"state_dict": modele.state_dict(), "in_channels": in_channels},
                CHECKPOINTS_DIR / "unet.pt",
            )
            print(f"  -> meilleur modèle sauvegardé (perte val {perte_val:.4f})")

    with open(REPORTS_DIR / "historique_entrainement.json", "w", encoding="utf-8") as f:
        json.dump(historique, f, indent=2)
    print(f"\nModèle sauvegardé dans {CHECKPOINTS_DIR / 'unet.pt'}")


if __name__ == "__main__":
    main()
