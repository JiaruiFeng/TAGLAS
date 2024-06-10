import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

allowable_features_map = {
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_chirality_dict": {
        "CHI_UNSPECIFIED": "unspecified",
        "CHI_TETRAHEDRAL_CW": "tetrahedral clockwise",
        "CHI_TETRAHEDRAL_CCW": "tetrahedral counter-clockwise",
        "CHI_OTHER": "other",
        "misc": "misc",
    },

    "possible_formal_charge_list": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        "misc",
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "misc",
    ],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": [
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
        "misc",
    ],
    "possible_bond_stereo_dict": {
        "STEREONONE": "none",
        "STEREOZ": "Z",
        "STEREOE": "E",
        "STEREOCIS": "CIS",
        "STEREOTRANS": "TRANS",
        "STEREOANY": "ANY",
    },
    "possible_is_conjugated_list": [False, True],
}


def ReorderCanonicalRankAtoms(mol):
    order = tuple(
        zip(
            *sorted(
                [(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))]
            )
        )
    )[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


def atom_to_feature(atom, chem_dict):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """

    atom_feature = [
        chem_dict[int(atom.GetAtomicNum())],
        "atomic number is " + str(atom.GetAtomicNum()),
        allowable_features_map["possible_chirality_dict"][
            str(atom.GetChiralTag())
        ]
        + " chirality",
        "degree of " + str(atom.GetTotalDegree()),
        "formal charge of " + str(atom.GetFormalCharge()),
        "num of hydrogen is " + str(atom.GetTotalNumHs()),
        "num of radical electrons is " + str(atom.GetNumRadicalElectrons()),
        "hybridization is " + str(atom.GetHybridization()),
        "is aromatic" if atom.GetIsAromatic() else "not aromatric",
        "is in ring" if atom.IsInRing() else "not in ring",
    ]
    return " , ".join(atom_feature)


def bond_to_feature(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        str(bond.GetBondType()) + " bond",
        "bond stereo is "
        + allowable_features_map["possible_bond_stereo_dict"][
            str(bond.GetStereo())
        ],
        "is conjugated" if bond.GetIsConjugated() else "not conjugated",
    ]
    return " , ".join(bond_feature)


def compute_cycle(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    return cycle_score


def smiles2graph(smiles_string, chem_dict, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    cycle_score = compute_cycle(mol)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom, chem_dict))

    # bonds
    edges_list = []
    edge_features_list = []
    bonds = mol.GetBonds()
    if len(bonds) == 0:
        edge_list = np.zeros((0, 2))
    else:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_list = np.array(edges_list)

    graph = dict()
    graph["edge_list"] = edge_list
    graph["edge_feat"] = edge_features_list
    graph["node_feat"] = atom_features_list
    graph["cycle"] = cycle_score

    return graph
