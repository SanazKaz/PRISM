# Copyright <2019> <University of Oxford>
# This code is licensed under MIT license (see LICENSE.txt for details)

import argparse, os, gzip
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
from rdkit.Chem import rdMolAlign
#################################################
#### Setting up the features to use in FeatureMap
fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
#    keep = ('Donor','Acceptor','NegIonizable','PosIonizable','Aromatic')



fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder',
        'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')
#################################################

def get_FeatureMapScore(small_m, large_m, score_mode=FeatMaps.FeatMapScoreMode.Best):
    featLists = []
    for m in [small_m, large_m]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = score_mode
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))
    return fm_score

def main(ref_file, prb_file, score_mode=FeatMaps.FeatMapScoreMode.Best, write=False, return_all=False):
         
    if type(ref_file) == str:
        if os.path.splitext(ref_file)[-1] == '.sdf':
            reflig = Chem.MolFromMolFile(ref_file, sanitize=True)
        elif os.path.splitext(ref_file)[-1] == '.mol2':
            reflig = Chem.MolFromMol2File(ref_file, sanitize=True)
    elif type(ref_file) == rdkit.Chem.rdchem.Mol:
        reflig = ref_file

    if type(prb_file) == str:
        if os.path.splitext(prb_file)[-1] == '.sdf':
            prb_mols = Chem.SDMolSupplier(prb_file, sanitize=True)
        elif os.path.splitext(prb_file)[-1] == '.gz':
            tmp = os.path.splitext(prb_file)[0]
            if os.path.splitext(tmp)[-1] == '.sdf':
                inf = gzip.open(prb_file)
                prb_mols = Chem.ForwardSDMolSupplier(inf, sanitize=True)
    elif type(prb_file) == rdkit.Chem.rdchem.Mol:
        prb_mols = [prb_file]
     
    try: reflig
    except NameError:
        raise ValueError("Incorrect file format for ref lig" )
    try: prb_mols
    except NameError:
        raise ValueError("Incorrect file format for prb lig" )

    if write: w = Chem.SDWriter("%s_SuCOS_score.sdf" % os.path.splitext(prb_file)[0])
    prb_mols = [x for x in prb_mols if x]
    
    for prb_mol in prb_mols:
        ##############################################
        ####### Feature map
        ##############################################
        
        fm_score = get_FeatureMapScore(reflig, prb_mol, score_mode)
        #fm_score = np.clip(fm_score, 0, 1) #Commented out, no longer necessary if using score_mode=Best (~Marc)
        ##############################################

        #tversky_ind = rdShapeHelpers.ShapeTverskyIndex(reflig, prb_mol, 1.0, 0.0)
        #SuCOS_score = 0.5*fm_score + 0.5*tversky_ind
        
        protrude_dist = rdShapeHelpers.ShapeProtrudeDist(reflig, prb_mol,
                allowReordering=False)
        #protrude_dist = np.clip(protrude_dist, 0, 1)
        SuCOS_score = 0.5*fm_score + 0.5*(1 - protrude_dist)

		#For high throughput consider commenting out print statements to deconvolute terminal (~Marc)
        print ("********************************")
        print ("SuCOS score:\t%f" % SuCOS_score)
        print ("Chem features:\t%f" % fm_score)
        print ("ShapeProtrudeDist:\t%f" % protrude_dist)
        print ("********************************")

        # prb_mol.SetProp("SuCOS_score", str(SuCOS_score))
        # prb_mol.SetProp("Volume_score", str(1 - protrude_dist))
        # prb_mol.SetProp("Feature_score", str(fm_score))
        if write:
            w.write(prb_mol)
    if return_all:
        return SuCOS_score, fm_score, (1 - protrude_dist)
    else:
        return SuCOS_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run SuCOS")
    parser.add_argument('--lig1', help='the smaller/reference ligand, in sdf or mol2 file\
                        format.')
    parser.add_argument('--lig2', help='the larger/query ligand(s), in sdf or .sdf.gz\
                        file format.')
    parser.add_argument('--write', action='store_true', default=False,
                        help='writes the SuCOS score into a sdf file with\
                        suffix _SuCOS_score.sdf')
    parser.add_argument('--return_all', action='store_true', default=False)
    parser.add_argument('--score_mode', choices=['all', 'closest', 'best'],
                        help='choose the scoring mode for the feature map,\
                        default is best.')

    args = parser.parse_args()

    ref_file = args.lig1
    prb_file = args.lig2

    if args.score_mode:
        print("mode specified:")
        if args.score_mode == 'all':
            print ("Feature maps scoring by all. WARNING: THIS IS NOT NORMALIZED")
            score_mode = FeatMaps.FeatMapScoreMode.All
        elif args.score_mode == 'closest':
            print ("Feature maps scoring by closest. WARNING: THIS IS NOT NORMALIZED")
            score_mode = FeatMaps.FeatMapScoreMode.Closest
        elif args.score_mode == 'best': #FeatMapScoreMode.Best normalizes SuCOS ##MARC 
            print ("Feature maps scoring by best. This is standard.")
            score_mode = FeatMaps.FeatMapScoreMode.Best
        else:
            print ("This is not an option")
    else:
        print("standard mode utilized:")
        print ("Feature maps scoring by best. This is standard.")
        score_mode = FeatMaps.FeatMapScoreMode.Best

    main(ref_file, prb_file, score_mode, args.write, args.return_all)
