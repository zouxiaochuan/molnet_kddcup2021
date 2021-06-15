import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import numpy as np
import torch


DIM_ATOM_CATE_FEAT = 16
DIM_ATOM_FLOAT_FEAT = 3
DIM_BOND_CATE_FEAT = 6
DIM_BOND_FLOAT_FEAT = 2

DICT_CHIRALTAG = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.ChiralType.values.values())
}

DICT_HYBRIDIZATION = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.HybridizationType.values.values())
}

DICT_BOND_DIR = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.BondDir.values.values())
}

DICT_BOND_TYPE = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.BondType.values.values())
}

DICT_BOND_STEREO = {
    v: i for i, v in enumerate(rdkit.Chem.rdchem.BondStereo.values.values())
}


def extract_mol_xyz(mol: rdkit.Chem.Mol):
    try:
        mol2 = rdkit.Chem.AddHs(mol)
        if rdkit.Chem.AllChem.EmbedMolecule(
                mol2, randomSeed=79112) != 0:
            failEmbed = True
            pass
        else:
            failEmbed = False
            pass
        pass
    except RuntimeError as e:
        print(e)
        failEmbed = True
        pass

    if failEmbed:
        return None
    else:
        while True:
            try:
                mmf_result = rdkit.Chem.AllChem.MMFFOptimizeMolecule(
                    mol2, maxIters=200)
            except rdkit.Chem.rdchem.KekulizeException:
                mol2 = rdkit.Chem.AddHs(mol)
                rdkit.Chem.AllChem.EmbedMolecule(mol2, randomSeed=79112)
                mmf_result = -1
                pass
            
            if mmf_result != 1:
                break
            pass
        if mmf_result == 1:
            raise RuntimeError(rdkit.Chem.MolToSmiles(mol))
        elif mmf_result == -1:
            while True:
                try:
                    uff_result = rdkit.Chem.AllChem.UFFOptimizeMolecule(
                        mol2, maxIters=200)
                except RuntimeError:
                    uff_result = 0
                    pass
                except rdkit.Chem.rdchem.KekulizeException:
                    mol2 = rdkit.Chem.AddHs(mol)
                    rdkit.Chem.AllChem.EmbedMolecule(mol2, randomSeed=79112)
                    uff_result = 0
                    pass
                
                if uff_result != 1:
                    break
                pass
            if uff_result != 0:
                raise RuntimeError(rdkit.Chem.MolToSmiles(mol))
            pass
        
        mol2 = rdkit.Chem.RemoveHs(mol2)

        xyz = mol2.GetConformer().GetPositions()
        return np.asarray(xyz, dtype='float32'), mol2, mmf_result
    pass


def extract_mol_multi_xyz(mol: rdkit.Chem.Mol, num):
    try:
        mol2 = rdkit.Chem.AddHs(mol)
        rdkit.Chem.AllChem.EmbedMultipleConfs(
            mol2, numConfs=num, randomSeed=79112)

        if mol2.GetNumConformers() == 0:
            failEmbed = True
            pass
        else:
            failEmbed = False
            pass
        pass
    except RuntimeError as e:
        print(e)
        failEmbed = True
        pass

    if failEmbed:
        return None
    else:
        try:
            mmf_result = rdkit.Chem.AllChem.MMFFOptimizeMolecule(
                mol2, maxIters=100)
            mol2 = rdkit.Chem.RemoveHs(mol2)
        except rdkit.Chem.rdchem.KekulizeException:
            mol2 = rdkit.Chem.AddHs(mol)
            rdkit.Chem.AllChem.EmbedMolecule(mol2, randomSeed=79112)
            mol2 = rdkit.Chem.RemoveHs(mol2)
            mmf_result = -1
            pass
            
        if mmf_result == 1:
            mmf_result = 0
        
        mmf_result = -1

        num_conf = mol2.GetNumConformers()
        num_atom = mol2.GetNumAtoms()
        xyz = np.zeros((num_conf, num_atom, 3), dtype='float32')

        for i in range(num_conf):
            xyz[i] = mol2.GetConformer(i).GetPositions()
            pass
        
        return xyz, mol2, mmf_result
    pass


def extract_mol_xy(mol: rdkit.Chem.Mol):
    rdkit.Chem.AllChem.Compute2DCoords(mol)
    xyz = np.asarray(
        mol.GetConformer().GetPositions(),
        dtype='float32')

    return xyz[:, :2]


def extract_atom_cate_feat(mol: rdkit.Chem.Mol):

    feat = np.zeros((mol.GetNumAtoms(), DIM_ATOM_CATE_FEAT), dtype='int32')
    
    for i in range(mol.GetNumAtoms()):
        atom: rdkit.Chem.Atom
        atom = mol.GetAtomWithIdx(i)

        feat[i, 0] = atom.GetAtomicNum() - 1
        feat[i, 1] = DICT_CHIRALTAG[atom.GetChiralTag()]
        feat[i, 2] = atom.GetTotalDegree()
        feat[i, 3] = atom.GetDegree()
        feat[i, 4] = atom.GetTotalNumHs()
        feat[i, 5] = atom.GetNumRadicalElectrons()
        feat[i, 6] = DICT_HYBRIDIZATION[atom.GetHybridization()]
        feat[i, 7] = int(atom.GetIsAromatic())
        feat[i, 8] = int(atom.IsInRing())
        feat[i, 9] = atom.GetExplicitValence()
        feat[i, 10] = atom.GetImplicitValence()
        feat[i, 11] = atom.GetIsotope()
        feat[i, 12] = atom.GetFormalCharge() + 5
        feat[i, 13] = int(atom.GetNoImplicit())
        feat[i, 14] = atom.GetNumExplicitHs()
        feat[i, 15] = atom.GetNumImplicitHs()
        
        pass

    return feat
    pass


def extract_atom_float_feat(mol: rdkit.Chem.Mol):
    # result = extract_mol_xyz(mol)
    # xyz_success = True
    # 
    # if result is None:
    #     xyz = np.zeros((mol.GetNumAtoms(), 3), dtype='float32')
    #     xyz_success = False
    #     mmf_result = -1
    #     pass
    # else:
    #     xyz, mol, mmf_result = result
    #     pass

    xy = extract_mol_xy(mol)

    feat = np.zeros((mol.GetNumAtoms(), DIM_ATOM_FLOAT_FEAT), dtype='float32')
    feat[:, :2] = xy
    
    for i, atom in enumerate(mol.GetAtoms()):
        feat[i, 2] = 1.0 / atom.GetMass()
        pass

    return feat
    pass


def extract_bond_feat(mol: rdkit.Chem.Mol):
    num_bonds = mol.GetNumBonds()
    if num_bonds == 0:
        bond_index = np.empty((2, 0), dtype='int32')
        bond_cate_feat = np.empty((0, DIM_BOND_CATE_FEAT), dtype='int32')
        bond_float_feat = np.empty((0, DIM_BOND_FLOAT_FEAT), dtype='float32')
        pass
    else:
        bond_index = np.zeros((2, num_bonds), dtype='int32')
        bond_cate_feat = np.zeros((num_bonds, DIM_BOND_CATE_FEAT),
                                  dtype='int32')
        bond_float_feat = np.zeros((num_bonds, DIM_BOND_FLOAT_FEAT),
                                   dtype='float32')

        for ib, bond in enumerate(mol.GetBonds()):
            bond: rdkit.Chem.Bond
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
        
            bond_index[0, ib] = i            
            bond_index[1, ib] = j

            bond_cate_feat[ib, 0] = DICT_BOND_DIR[bond.GetBondDir()]
            bond_cate_feat[ib, 1] = DICT_BOND_TYPE[bond.GetBondType()]
            bond_cate_feat[ib, 2] = int(bond.GetIsAromatic())
            bond_cate_feat[ib, 3] = int(bond.GetIsConjugated())
            bond_cate_feat[ib, 4] = int(bond.IsInRing())
            bond_cate_feat[ib, 5] = DICT_BOND_STEREO[bond.GetStereo()]

            bond_float_feat[ib, 0] = bond.GetValenceContrib(
                bond.GetBeginAtom())
            bond_float_feat[ib, 1] = bond.GetValenceContrib(
                bond.GetEndAtom())
            pass
        pass
    
    return bond_index, bond_cate_feat, bond_float_feat


def extract_mol_feat_cate(mol: rdkit.Chem.Mol):
    fea = np.zeros(4, dtype='float32')
    fea[0] = mol.GetNumAtoms()
    fea[1] = mol.GetNumBonds()
    fea[2] = mol.GetNumHeavyAtoms()
    ringinfo = mol.GetRingInfo()
    fea[3] = ringinfo.NumRings()
    return fea


def extrac_mol_fingerprint(mol: rdkit.Chem.Mol):

    fp = np.asarray(rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3))

    return np.argwhere(fp).flatten().astype('int32')


def calc_bond_spatial_feat(
        bond_index,
        atom_xyz):
    shift = atom_xyz[bond_index[0]] - atom_xyz[bond_index[1]]
    
    dist = torch.norm(shift, dim=-1, keepdim=True)
    angle = torch.div(shift, dist + 1e-12)
    
    return torch.cat((dist, angle), dim=-1)
    # return torch.zeros((bond_index.shape[1], 4))
    pass
