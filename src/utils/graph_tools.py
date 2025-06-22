# cSpell:disable
from typing import Any, Dict, List

import torch
import torch_geometric
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, GraphDescriptors, Descriptors
from torch_geometric.data import Data

x_map: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
    "is_in_ring": [False, True],
}

def add_global_features(global_features: List[str], graph: Data) -> Data:
    g = []
    smiles = graph.smiles
    mol = Chem.MolFromSmiles(smiles)
    if "CalcKappa1" in global_features:
        try: 
            g.append(float(rdMolDescriptors.CalcKappa1(mol)))
        except:
            g.append(0.0)
            print("CalcKappa1 failed for", smiles)
    if "CalcKappa2" in global_features:
        try:
            g.append(float(rdMolDescriptors.CalcKappa2(mol)))
        except:
            g.append(0.0)
            print("CalcKappa2 failed for", smiles)
    if "CalcKappa3" in global_features:
        try:
            g.append(float(rdMolDescriptors.CalcKappa3(mol)))
        except:
            g.append(0.0)
            print("CalcKappa3 failed for", smiles)
    if "CalcLabuteASA" in global_features:
        try:
            g.append(float(rdMolDescriptors.CalcLabuteASA(mol)))
        except:
            g.append(0.0)
            print("CalcLabuteASA failed for", smiles)
    if "Chi0" in global_features:
        try:
            g.append(float(GraphDescriptors.Chi0(mol)))
        except:
            g.append(0.0)
            print("Chi0 failed for", smiles)
    if "Chi1" in global_features:
        try:
            g.append(float(GraphDescriptors.Chi1(mol)))
        except:
            g.append(0.0)
            print("Chi1 failed for", smiles)
    if "HeavyAtomMolWt" in global_features:
        try:
            g.append(float(Descriptors.HeavyAtomMolWt(mol)))
        except:
            g.append(0.0)
            print("HeavyAtomMolWt failed for", smiles)
    if "ExactMolWt" in global_features:
        try:
            g.append(float(Descriptors.ExactMolWt(mol)))
        except:
            g.append(0.0)
            print("ExactMolWt failed for", smiles)
    if "MolLogP" in global_features:
        try:
            g.append(float(Descriptors.MolLogP(mol)))
        except:
            g.append(0.0)
            print("MolLogP failed for", smiles)
    if "CalcFractionCSP3" in global_features:
        try:
            g.append(float(rdMolDescriptors.CalcFractionCSP3(mol)))
        except:
            g.append(0.0)
            print("CalcFractionCSP3 failed for", smiles)
    if "CalcHallKierAlpha" in global_features:
        try:
            g.append(float(rdMolDescriptors.CalcHallKierAlpha(mol)))
        except:
            g.append(0.0)
            print("CalcHallKierAlpha failed for", smiles)
    if "PEOE_VSA9" in global_features:
        try:
            g.append(float(Chem.MolSurf.PEOE_VSA9(mol)))
        except:
            g.append(0.0)
            print("PEOE_VSA9 failed for", smiles)
    if "SlogP_VSA1" in global_features:
        try:
            g.append(float(Chem.MolSurf.SlogP_VSA1(mol)))
        except:
            g.append(0.0)
            print("SlogP_VSA1 failed for", smiles)
    if "EState_VSA2" in global_features:
        try:
            g.append(float(Chem.EState.EState_VSA.EState_VSA2(mol)))
        except:
            g.append(0.0)
            print("EState_VSA2 failed for", smiles)
    if "MaxAbsEStateIndex" in global_features:
        try:
            g.append(float(Chem.EState.EState.MaxAbsEStateIndex(mol)))
        except:
            g.append(0.0)
            print("MaxAbsEStateIndex failed for", smiles)
    graph.g = torch.tensor(g, dtype=torch.float).view(1, -1)
    return graph

def from_rdmol(
    mol: Any,
    node_features: List[str] = ['atomic_num', 'chirality', 'degree', 'formal_charge', 
                              'num_hs', 'hybridization', 'is_aromatic', 'is_in_ring'],
    edge_features: List[str] = ['bond_type', 'stereo', 'is_conjugated', 'is_in_ring']
) -> "torch_geometric.data.Data":  # type: ignore
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
        node_features (List[str], optional): Node features to include in the graph.
        edge_features (List[str], optional): Edge features to include in the graph.
    """
    assert isinstance(mol, Chem.Mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():  # type: ignore
        row: List[int] = []
        for feature in node_features:
            if feature == 'atomic_num':
                row.append(1 + x_map["atomic_num"].index(atom.GetAtomicNum()))
            elif feature == 'chirality':
                row.append(1 + x_map["chirality"].index(str(atom.GetChiralTag())))
            elif feature == 'degree':
                row.append(1 + x_map["degree"].index(atom.GetTotalDegree()))
            elif feature == 'formal_charge':
                row.append(1 + x_map["formal_charge"].index(atom.GetFormalCharge()))
            elif feature == 'num_hs':
                row.append(1 + x_map["num_hs"].index(atom.GetTotalNumHs()))
            elif feature == 'num_radical_electrons':
                row.append(1 + x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons()))
            elif feature == 'hybridization':
                row.append(1 + x_map["hybridization"].index(str(atom.GetHybridization())))
            elif feature == 'is_aromatic':
                row.append(1 + x_map["is_aromatic"].index(atom.GetIsAromatic()))
            elif feature == 'is_in_ring':
                row.append(1 + x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, len(node_features))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        for feature in edge_features:
            if feature == 'bond_type':
                e.append(1 + e_map["bond_type"].index(str(bond.GetBondType())))
            elif feature == 'stereo':
                e.append(1 + e_map["stereo"].index(str(bond.GetStereo())))
            elif feature == 'is_conjugated':
                e.append(1 + e_map["is_conjugated"].index(bond.GetIsConjugated()))
            elif feature == 'is_in_ring':
                e.append(1 + e_map["is_in_ring"].index(bond.IsInRing()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(
        -1, len(edge_features)
    )  # 2nd number in view is edge dim

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
    node_features: List[str] = ['atomic_num', 'chirality', 'degree', 'formal_charge', 
                              'num_hs', 'hybridization', 'is_aromatic', 'is_in_ring'],
    edge_features: List[str] = ['bond_type', 'stereo', 'is_conjugated', 'is_in_ring']
) -> "torch_geometric.data.Data":  # type: ignore
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
        node_features (List[str], optional): Node features to include in the graph.
        edge_features (List[str], optional): Edge features to include in the graph.
    """

    RDLogger.DisableLog("rdApp.*")  # type: ignore
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    data = from_rdmol(mol, node_features=node_features, edge_features=edge_features)
    data.smiles = smiles
    return data


# Return functions


def to_rdmol(
    data: "torch_geometric.data.Data",  # type: ignore
    kekulize: bool = False,
) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a
    :class:`rdkit.Chem.Mol` instance.

    Args:
        data (torch_geometric.data.Data): The molecular graph data.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    mol = Chem.RWMol()

    assert data.x is not None
    assert data.num_nodes is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    for i in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(data.x[i, 1])])
        atom.SetFormalCharge(x_map["formal_charge"][int(data.x[i, 3])])
        atom.SetNumExplicitHs(x_map["num_hs"][int(data.x[i, 4])])
        atom.SetNumRadicalElectrons(x_map["num_radical_electrons"][int(data.x[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(data.x[i, 6])])
        atom.SetIsAromatic(bool(data.x[i, 7]))
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[int(data.edge_attr[i, 0])]
        mol.AddBond(src, dst, bond_type)

        # Set stereochemistry:
        stereo = Chem.rdchem.BondStereo.values[int(data.edge_attr[i, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        # Set conjugation:
        is_conjugated = bool(data.edge_attr[i, 2])
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return mol


def to_smiles(
    data: "torch_geometric.data.Data",  # type: ignore
    kekulize: bool = False,
) -> str:
    """Converts a :class:`torch_geometric.data.Data` instance to a SMILES
    string.

    Args:
        data (torch_geometric.data.Data): The molecular graph.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    mol = to_rdmol(data, kekulize=kekulize)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
