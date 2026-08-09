import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Geometry import Point3D
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from tqdm import tqdm

try:
    from data_processing.sgdataset import SizeGroupedDataset
except ModuleNotFoundError as exc:
    if exc.name != "data_processing":
        raise
    # Support direct execution as well as
    # `python -m data_processing.prepare_rgd1_chembl_extended`.
    from sgdataset import SizeGroupedDataset


# Keep stereo perception identical to app/utils.py.  Training stereo is assigned
# from endpoint coordinates below; inference in the app uses SMARTS stereo tags.
Chem.SetUseLegacyStereoPerception(True)

# Model bond encoding, shared with app/utils.py:
# 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic, 5-6=E/Z,
# 7-8=tetrahedral chirality.
BOND_TYPES = {
    BT.SINGLE: 1,
    BT.DOUBLE: 2,
    BT.TRIPLE: 3,
    BT.AROMATIC: 4,
}
SOURCE_BOND_TYPES = {
    1.0: BT.SINGLE,
    1.5: BT.AROMATIC,
    2.0: BT.DOUBLE,
    3.0: BT.TRIPLE,
    4.0: BT.AROMATIC,
}
CHI_BONDS = (7, 8)
EZ_BONDS = {
    Chem.BondStereo.STEREOE: 5,
    Chem.BondStereo.STEREOZ: 6,
}


def build_rdkit_mol(numbers, coords, bond_mat, *, connectivity_only=False):
    """Build RDKit molecule from atomic numbers, coordinates, and adjacency matrix."""
    mol = Chem.RWMol()
    for num in numbers:
        mol.AddAtom(Chem.Atom(int(num)))

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            value = float(bond_mat[i, j])
            if value == 0.0:
                continue
            if value not in SOURCE_BOND_TYPES:
                raise ValueError(f"Unsupported source bond order {value} at ({i}, {j})")

            bond_type = BT.SINGLE if connectivity_only else SOURCE_BOND_TYPES[value]
            mol.AddBond(i, j, bond_type)
            if bond_type == BT.AROMATIC:
                mol.GetAtomWithIdx(i).SetIsAromatic(True)
                mol.GetAtomWithIdx(j).SetIsAromatic(True)

    mol = mol.GetMol()
    conf = Chem.Conformer(len(numbers))
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(conf, assignId=True)
    return mol


def _kekulize_mol(mol, numbers, coords, bond_mat, charge):
    """Return a molecule with explicit single/double aromatic bond orders.

    Most source matrices can be kekulized directly.  A small number lack the
    atom-level formal charges RDKit needs (for example, N-oxides).  For those,
    infer a valid explicit bond-order/charge assignment from connectivity and
    total charge instead of silently truncating aromatic order 1.5 to 1.
    """
    kekulized = Chem.Mol(mol)
    try:
        # Some source records do not carry the atom-level formal charges needed
        # for direct kekulization. Suppress RDKit's expected error log here; the
        # fallback below reports the condition once through Python warnings.
        with rdBase.BlockLogs():
            Chem.Kekulize(kekulized, clearAromaticFlags=True)
        return kekulized
    except (RuntimeError, ValueError):
        warnings.warn(
            "Direct aromatic kekulization failed for at least one endpoint; "
            "falling back to RDKit bond-order inference.",
            RuntimeWarning,
            stacklevel=2,
        )

    errors = []
    for allow_charged_fragments in (True, False):
        inferred = build_rdkit_mol(
            numbers,
            coords,
            bond_mat,
            connectivity_only=True,
        )
        try:
            rdDetermineBonds.DetermineBondOrders(
                inferred,
                charge=int(charge),
                allowChargedFragments=allow_charged_fragments,
                embedChiral=False,
            )
            return inferred
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    raise ValueError(
        "Could not produce a kekulized bond assignment from the source matrix: "
        + " | ".join(errors)
    )


def _bond_matrix_from_mol(mol):
    """Convert RDKit bonds to the model's integer bond encoding."""
    bmat = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.int64)
    for bond in mol.GetBonds():
        try:
            value = BOND_TYPES[bond.GetBondType()]
        except KeyError as exc:
            raise ValueError(f"Unsupported RDKit bond type: {bond.GetBondType()}") from exc
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bmat[i, j] = value
        bmat[j, i] = value
    return bmat


def add_stereo_bonds(mol, chi_bonds, ez_bonds, bmat, from_3D=True):
    """Add the same E/Z and tetrahedral pseudo-edges used by app/utils.py."""
    result = []
    if from_3D:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [nbr.GetIdx() for nbr in atom_1.GetNeighbors() if nbr.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [nbr.GetIdx() for nbr in atom_2.GetNeighbors() if nbr.GetIdx() not in {idx_1, idx_4}]

            inv_stereo = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])

            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]), (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]), (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]), (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp('_CIPCode'):
            chirality = bond.GetBeginAtom().GetProp('_CIPCode')
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a, b, c = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d = sorted_neighbors[-1]
                result.extend([
                    (a, d, chi_bonds[0]), (b, d, chi_bonds[0]), (c, d, chi_bonds[0]),
                    (d, a, chi_bonds[0]), (d, b, chi_bonds[0]), (d, c, chi_bonds[0]),
                    (b, a, chi_bonds[1]), (c, b, chi_bonds[1]), (a, c, chi_bonds[1])
                ])

    for i, j, value in result:
        # Stereo is represented by additional pseudo-edges, never by replacing
        # a physical bond.
        if bmat[i, j] == 0:
            bmat[i, j] = value
    return bmat


def prepare_endpoint_bonds(
    numbers,
    coords,
    source_bmat,
    charge,
    *,
    kekulize=True,
    add_stereo=True,
):
    """Prepare one endpoint using app-compatible bond and stereo encodings.

    Unlike SMARTS inference, training has endpoint geometries, so stereo tags
    are deliberately assigned from 3D coordinates.
    """
    mol = build_rdkit_mol(numbers, coords, source_bmat)
    if kekulize:
        mol = _kekulize_mol(mol, numbers, coords, source_bmat, charge)

    bmat = _bond_matrix_from_mol(mol)
    if add_stereo:
        bmat = add_stereo_bonds(
            mol,
            CHI_BONDS,
            EZ_BONDS,
            bmat,
            from_3D=True,
        )
    return bmat


def process_dataset(group, *, kekulize=True, add_stereo=True):
    mols = []

    for idx in range(len(group["_id"])):
        numbers = torch.from_numpy(group["numbers"][idx]).to(torch.uint8)
        r_coord = torch.from_numpy(group["r_coord"][idx])
        p_coord = torch.from_numpy(group["p_coord"][idx])
        ts_coord = torch.from_numpy(group["ts_coord"][idx])
        mol_id = str(group["_id"][idx])

        charge_value = int(group["charge"][idx])
        charges = torch.full_like(numbers, charge_value, dtype=torch.int8)

        if ts_coord.shape[0] != numbers.shape[0]:
            print(f"[WARNING] Skipping molecule {mol_id} due to shape mismatch: "
                  f"ts_coord {ts_coord.shape[0]} vs numbers {numbers.shape[0]}")
            continue

        bmat_r = torch.from_numpy(prepare_endpoint_bonds(
            group["numbers"][idx],
            group["r_coord"][idx],
            group["bmat_r"][idx],
            charge_value,
            kekulize=kekulize,
            add_stereo=add_stereo,
        ))
        bmat_p = torch.from_numpy(prepare_endpoint_bonds(
            group["numbers"][idx],
            group["p_coord"][idx],
            group["bmat_p"][idx],
            charge_value,
            kekulize=kekulize,
            add_stereo=add_stereo,
        ))

        edge_index = (bmat_r + bmat_p).nonzero().contiguous().T
        edge_attr = torch.stack([bmat_r, bmat_p], dim=-1)[edge_index[0], edge_index[1]].to(torch.uint8)

        data = Data(
            numbers=numbers,
            charges=charges,
            ts_coord=ts_coord,
            r_coord=r_coord,
            p_coord=p_coord,
            id=mol_id,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=numbers.shape[0]
        )
        mols.append(data)

    return mols


def save_pyg_dataset(pyg_batch, save_path):
    """Save a list of PyG Data objects to disk as a batched tensor."""
    batch = collate(pyg_batch[0].__class__, pyg_batch, increment=False, add_batch=False)
    torch.save(batch[:2], save_path)  # save (data, slices)


def main(args):
    ds = SizeGroupedDataset.from_h5(args.h5_file)

    save_data_path = Path(args.save_data_folder)
    save_data_path.mkdir(parents=True, exist_ok=True)
    save_processed_path = save_data_path / "processed"
    save_processed_path.mkdir(parents=True, exist_ok=True)

    split = ds.random_split(0.96, 0.02, 0.02)

    print(f"Split sizes: train={len(split[0])}, val={len(split[1])}, test={len(split[2])}")

    for s, sname in zip(split, ["train", "val", "test"]):
        pyg_mols = []

        for n in tqdm(ds.keys(), desc=f"Processing {sname} set"):
            group = s[n]
            group = {key: group[key][:] for key in group.keys()}
            _pyg_mols = process_dataset(
                group,
                kekulize=args.kekulize,
                add_stereo=args.add_stereo,
            )
            pyg_mols.extend(_pyg_mols)

        save_pyg_dataset(pyg_mols, save_processed_path / f"{sname}_h.pt")
        print(f"Saved {len(pyg_mols)} molecules to {sname}_h.pt")

    print(f"All splits saved to: {save_processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare RGD1-ChEMBL-extended data for RitS training"
    )
    parser.add_argument("--h5_file", type=str, required=True,
                        help="Path to the h5 dataset file")
    parser.add_argument("--save_data_folder", type=str, required=True,
                        help="Directory to save the resulting PyG datasets")
    parser.add_argument(
        "--kekulize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use explicit single/double bonds for aromatic systems (default: true)",
    )
    parser.add_argument(
        "--add-stereo",
        "--add_stereo",
        dest="add_stereo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add 3D-derived E/Z and tetrahedral pseudo-edges (default: true)",
    )
    args = parser.parse_args()

    main(args)
