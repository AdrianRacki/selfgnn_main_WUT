# cSpell:disable
from typing import Any

import torch
import torch_geometric
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors
from torch_geometric.data import Data

x_map: dict[str, list[Any]] = {
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

e_map: dict[str, list[Any]] = {
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


def add_graph_mol_mapping(graph: Data) -> Data:
    smiles = graph.smiles
    mol_smiles_list = smiles.split(".")
    mapping = []
    for mol_idx, smi in enumerate(mol_smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles("")
        mapping.extend([mol_idx] * mol.GetNumAtoms())
    graph.map = torch.tensor(mapping, dtype=torch.long)
    return graph


def split_graph_to_mols(graph: Data) -> list[torch.Tensor]:
    mol_map = graph.map  # [num_nodes], each value is molecule index
    num_mols = int(mol_map.max().item()) + 1
    return [graph.x[mol_map == mol_idx] for mol_idx in range(num_mols)]


def split_graph_by_map(graph: Data) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split node features and batch indices by molecule mapping.

    Returns a list of (x_mol, batch_mol) tuples, one per molecule.
    """
    num_mols = int(graph.map.max().item()) + 1
    return [(graph.x[graph.map == mol_idx], graph.batch[graph.map == mol_idx]) for mol_idx in range(num_mols)]


def split_global_to_mols(graph: Data) -> list[torch.Tensor]:
    num_mols = graph.g.size(0)
    return [graph.g[mol_idx].unsqueeze(0) for mol_idx in range(num_mols)]


def add_global_features(global_features: list[str], graph: Data, separate_for_mols: bool = True) -> Data:
    def _compute(mol, smi):
        g = []
        if "CalcKappa1" in global_features:
            try:
                g.append(float(rdMolDescriptors.CalcKappa1(mol)))
            except:
                g.append(0.0)
                print("CalcKappa1 failed for", smi)
        if "CalcKappa2" in global_features:
            try:
                g.append(float(rdMolDescriptors.CalcKappa2(mol)))
            except:
                g.append(0.0)
                print("CalcKappa2 failed for", smi)
        if "CalcKappa3" in global_features:
            try:
                g.append(float(rdMolDescriptors.CalcKappa3(mol)))
            except:
                g.append(0.0)
                print("CalcKappa3 failed for", smi)
        if "CalcLabuteASA" in global_features:
            try:
                g.append(float(rdMolDescriptors.CalcLabuteASA(mol)))
            except:
                g.append(0.0)
                print("CalcLabuteASA failed for", smi)
        if "Chi0" in global_features:
            try:
                g.append(float(GraphDescriptors.Chi0(mol)))
            except:
                g.append(0.0)
                print("Chi0 failed for", smi)
        if "Chi1" in global_features:
            try:
                g.append(float(GraphDescriptors.Chi1(mol)))
            except:
                g.append(0.0)
                print("Chi1 failed for", smi)
        if "HeavyAtomMolWt" in global_features:
            try:
                g.append(float(Descriptors.HeavyAtomMolWt(mol)))
            except:
                g.append(0.0)
                print("HeavyAtomMolWt failed for", smi)
        if "ExactMolWt" in global_features:
            try:
                g.append(float(Descriptors.ExactMolWt(mol)))
            except:
                g.append(0.0)
                print("ExactMolWt failed for", smi)
        if "MolLogP" in global_features:
            try:
                g.append(float(Descriptors.MolLogP(mol)))
            except:
                g.append(0.0)
                print("MolLogP failed for", smi)
        if "CalcFractionCSP3" in global_features:
            try:
                g.append(float(rdMolDescriptors.CalcFractionCSP3(mol)))
            except:
                g.append(0.0)
                print("CalcFractionCSP3 failed for", smi)
        if "CalcHallKierAlpha" in global_features:
            try:
                g.append(float(rdMolDescriptors.CalcHallKierAlpha(mol)))
            except:
                g.append(0.0)
                print("CalcHallKierAlpha failed for", smi)
        if "PEOE_VSA9" in global_features:
            try:
                g.append(float(Chem.MolSurf.PEOE_VSA9(mol)))
            except:
                g.append(0.0)
                print("PEOE_VSA9 failed for", smi)
        if "SlogP_VSA1" in global_features:
            try:
                g.append(float(Chem.MolSurf.SlogP_VSA1(mol)))
            except:
                g.append(0.0)
                print("SlogP_VSA1 failed for", smi)
        if "EState_VSA2" in global_features:
            try:
                g.append(float(Chem.EState.EState_VSA.EState_VSA2(mol)))
            except:
                g.append(0.0)
                print("EState_VSA2 failed for", smi)
        if "MaxAbsEStateIndex" in global_features:
            try:
                g.append(float(Chem.EState.EState.MaxAbsEStateIndex(mol)))
            except:
                g.append(0.0)
                print("MaxAbsEStateIndex failed for", smi)
        return g

    smiles = graph.smiles
    if separate_for_mols:
        mol_smiles_list = smiles.split(".")
        all_g = []
        for smi in mol_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = Chem.MolFromSmiles("")
            all_g.append(_compute(mol, smi))
        graph.g = torch.tensor(all_g, dtype=torch.float)  # (n_mols, n_features)
    else:
        mol = Chem.MolFromSmiles(smiles)
        graph.g = torch.tensor(_compute(mol, smiles), dtype=torch.float).view(1, -1)
    return graph


def from_rdmol(
    mol: Any,
    node_features: list[str] = [
        "atomic_num",
        "chirality",
        "degree",
        "formal_charge",
        "num_hs",
        "hybridization",
        "is_aromatic",
        "is_in_ring",
    ],
    edge_features: list[str] = ["bond_type", "stereo", "is_conjugated", "is_in_ring"],
) -> "torch_geometric.data.Data":  # type: ignore
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
        node_features (List[str], optional): Node features to include in the graph.
        edge_features (List[str], optional): Edge features to include in the graph.
    """
    assert isinstance(mol, Chem.Mol)
    pt = Chem.GetPeriodicTable()
    _nx_features = {"closness_centrality", "betweenness_centrality", "harmonic_centrality", "page_rank"}
    _conformer_features = {"sasa", "normalized_distance_to_centroid"}
    if any(f in node_features for f in _nx_features):
        import networkx as nx

        nx_frag_indices = Chem.GetMolFrags(mol)
        nx_frag_mols = Chem.GetMolFrags(mol, asMols=True)

        closeness_map: dict[int, float] = {}
        betweenness_map: dict[int, float] = {}
        harmonic_map: dict[int, float] = {}
        pagerank_map: dict[int, float] = {}
        for orig_indices, frag_mol in zip(nx_frag_indices, nx_frag_mols):
            g_nx = nx.Graph()
            g_nx.add_nodes_from(range(frag_mol.GetNumAtoms()))
            for bond in frag_mol.GetBonds():
                g_nx.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if "closness_centrality" in node_features:
                cc = nx.closeness_centrality(g_nx)
                for frag_idx, orig_idx in enumerate(orig_indices):
                    closeness_map[orig_idx] = cc.get(frag_idx, 0.0)
            if "betweenness_centrality" in node_features:
                bc = nx.betweenness_centrality(g_nx)
                for frag_idx, orig_idx in enumerate(orig_indices):
                    betweenness_map[orig_idx] = bc.get(frag_idx, 0.0)
            if "harmonic_centrality" in node_features:
                hc = nx.harmonic_centrality(g_nx)
                for frag_idx, orig_idx in enumerate(orig_indices):
                    harmonic_map[orig_idx] = hc.get(frag_idx, 0.0)
            if "page_rank" in node_features:
                pr = nx.pagerank(g_nx)
                for frag_idx, orig_idx in enumerate(orig_indices):
                    pagerank_map[orig_idx] = pr.get(frag_idx, 0.0)
    if any(f in node_features for f in _conformer_features):
        from rdkit.Chem import AllChem, rdFreeSASA

        frag_atom_indices = Chem.GetMolFrags(mol)
        frag_mols = Chem.GetMolFrags(mol, asMols=True)

        sasa_map: dict[int, float] = {}
        dist_map: dict[int, float] = {}
        for orig_indices, frag_mol in zip(frag_atom_indices, frag_mols):
            frag_hs = Chem.AddHs(frag_mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            res = AllChem.EmbedMolecule(frag_hs, params)
            if res == -1:
                for orig_idx in orig_indices:
                    sasa_map[orig_idx] = 0.0
                    dist_map[orig_idx] = 0.0
                continue

            if "sasa" in node_features:
                radii = rdFreeSASA.classifyAtoms(frag_hs)
                rdFreeSASA.CalcSASA(frag_hs, radii)

            if "normalized_distance_to_centroid" in node_features:
                conf = frag_hs.GetConformer()
                n_heavy = frag_mol.GetNumAtoms()
                positions = [conf.GetAtomPosition(i) for i in range(n_heavy)]
                cx = sum(p.x for p in positions) / n_heavy
                cy = sum(p.y for p in positions) / n_heavy
                cz = sum(p.z for p in positions) / n_heavy
                dists = [((p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2) ** 0.5 for p in positions]
                max_dist = max(dists) if max(dists) > 0 else 1.0
                for frag_idx, orig_idx in enumerate(orig_indices):
                    dist_map[orig_idx] = dists[frag_idx] / max_dist

            for frag_idx, orig_idx in enumerate(orig_indices):
                if "sasa" in node_features:
                    sasa_map[orig_idx] = float(frag_hs.GetAtomWithIdx(frag_idx).GetDoubleProp("SASA"))

    xs: list[list[int]] = []
    for atom in mol.GetAtoms():  # type: ignore
        row: list[int] = []
        for feature in node_features:
            if feature == "atomic_num":
                row.append(1 + x_map["atomic_num"].index(atom.GetAtomicNum()))
            elif feature == "chirality":
                row.append(1 + x_map["chirality"].index(str(atom.GetChiralTag())))
            elif feature == "degree":
                row.append(1 + x_map["degree"].index(atom.GetTotalDegree()))
            elif feature == "formal_charge":
                row.append(1 + x_map["formal_charge"].index(atom.GetFormalCharge()))
            elif feature == "num_hs":
                row.append(1 + x_map["num_hs"].index(atom.GetTotalNumHs()))
            elif feature == "num_radical_electrons":
                row.append(1 + x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons()))
            elif feature == "hybridization":
                row.append(1 + x_map["hybridization"].index(str(atom.GetHybridization())))
            elif feature == "is_aromatic":
                row.append(1 + x_map["is_aromatic"].index(atom.GetIsAromatic()))
            elif feature == "is_in_ring":
                row.append(1 + x_map["is_in_ring"].index(atom.IsInRing()))
            elif feature == "logp_contrib":
                logp_contrib, _ = _get_logp_mr_contrib(mol, atom.GetIdx())  # type: ignore
                row.append(logp_contrib)
            elif feature == "mr_contrib":
                _, mr_contrib = _get_logp_mr_contrib(mol, atom.GetIdx())  # type: ignore
                row.append(mr_contrib)
            elif feature == "ring_size":
                ring_info = mol.GetRingInfo()
                ring_size = ring_info.AtomRingSizes(atom.GetIdx())
                if ring_size:
                    row.append(1 + ring_size[0])  # type: ignore
                else:
                    row.append(0)  # type: ignore
            elif feature == "vdw_radius":
                row.append(pt.GetRvdw(atom.GetAtomicNum()))  # type: ignore
            elif feature == "atomic_mass":
                row.append(pt.GetAtomicWeight(atom.GetAtomicNum()))  # type: ignore
            elif feature == "cov_radius":
                row.append(pt.GetRcovalent(atom.GetAtomicNum()))  # type: ignore
            elif feature == "sasa":
                row.append(sasa_map.get(atom.GetIdx(), 0.0))
            elif feature == "normalized_distance_to_centroid":
                row.append(dist_map.get(atom.GetIdx(), 0.0))
            elif feature == "closness_centrality":
                row.append(closeness_map.get(atom.GetIdx(), 0.0))
            elif feature == "betweenness_centrality":
                row.append(betweenness_map.get(atom.GetIdx(), 0.0))
            elif feature == "harmonic_centrality":
                row.append(harmonic_map.get(atom.GetIdx(), 0.0))
            elif feature == "page_rank":
                row.append(pagerank_map.get(atom.GetIdx(), 0.0))

        xs.append(row)

    x = torch.tensor(xs, dtype=torch.float32).view(-1, len(node_features))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        for feature in edge_features:
            if feature == "bond_type":
                e.append(1 + e_map["bond_type"].index(str(bond.GetBondType())))
            elif feature == "stereo":
                e.append(1 + e_map["stereo"].index(str(bond.GetStereo())))
            elif feature == "is_conjugated":
                e.append(1 + e_map["is_conjugated"].index(bond.GetIsConjugated()))
            elif feature == "is_in_ring":
                e.append(1 + e_map["is_in_ring"].index(bond.IsInRing()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, len(edge_features))  # 2nd number in view is edge dim

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _get_logp_mr_contrib(mol: Any, atom_idx: int) -> tuple[float, float]:
    """Get the contribution of a specific atom to the molecule's logP and MR."""
    try:
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)  # type: ignore
        logp_contrib, mr_contrib = contribs[atom_idx]
        return float(logp_contrib), float(mr_contrib)
    except:
        print(f"Failed to compute logP/MR contrib for atom {atom_idx} in molecule {Chem.MolToSmiles(mol)}")
        return 0.0, 0.0


def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
    node_features: list[str] = [
        "atomic_num",
        "chirality",
        "degree",
        "formal_charge",
        "num_hs",
        "hybridization",
        "is_aromatic",
        "is_in_ring",
        "logp_contrib",  # float
        "mr_contrib",  # float
        "ring_size",
        "vdw_radius",  # float
        "cov_radius",  # float
        "atomic_mass",  # float
        "sasa",  # float
        "normalized_distance_to_centroid",  # float
        "closness_centrality",  # float
        "betweenness_centrality",  # float
        "harmonic_centrality",  # float
        "page_rank",  # float
    ],
    edge_features: list[str] = ["bond_type", "stereo", "is_conjugated", "is_in_ring"],
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
